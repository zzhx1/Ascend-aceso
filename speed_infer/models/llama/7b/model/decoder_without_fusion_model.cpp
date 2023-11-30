/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "decoder_without_fusion_model.h"
#include "models/llama/13b/layer/parallel_layer.h"

namespace atb_speed {
namespace llama_7b {
const int WEIGHT_COUNT_PER_LAYER = 9;
const int INPUT_TENSOR_COUNT_BEFORE_KEY = 5;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY
};

void DecoderWithoutFusionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    ATB_LOG(INFO) << "ChatGlm2DecoderModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize:" << rankSize;
}

DecoderWithoutFusionModel::DecoderWithoutFusionModel(const std::string &param)
    : Model("DecoderWithoutFusionModel", param)
{
    param_.FromString(param);
}

DecoderWithoutFusionModel::~DecoderWithoutFusionModel() = default;

uint32_t DecoderWithoutFusionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DecoderWithoutFusionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status DecoderWithoutFusionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                  std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter DecoderWithoutFusionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const atb::TensorDesc &hiddenStateDesc = inTensorDescs.at(IN_TENSOR_HIDDENSTATES);
    const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_TENSOR_PAST_KEY);
    const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_TENSOR_PAST_KEY + param_.layerNum);

    outTensorDescs.at(0) = hiddenStateDesc;
    for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
        outTensorDescs.at(1 + keyId) = keyTensorDesc;
        outTensorDescs.at(1 + keyId).shape.dims[1] += 1;
    }
    for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
        outTensorDescs.at(1 + param_.layerNum + valueId) = valueTensorDesc;
        outTensorDescs.at(1 + param_.layerNum + valueId).shape.dims[1] += 1;
    }

    return atb::NO_ERROR;
}

void DecoderWithoutFusionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter DecoderWithoutFusionModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(INPUT_TENSOR_COUNT_BEFORE_KEY + param_.layerNum * 2);
    graph_.outTensors.resize(OUTPUT_TENSOR_COUNT_BEFORE_KEY + param_.layerNum * 2);

    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    atb::Operation *op = nullptr;
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::llama_13b::ParallelLayerParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.model = "llama13b";
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;
        atb_speed::llama_13b::DecoderParallelLayer(modelParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY + layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY + layerId + param_.layerNum);

        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(layerId), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        }

        firstInTensor = layerNode.outTensors.at(0);
    }
}
} // namespace llama_7b
} // namespace atb_speed