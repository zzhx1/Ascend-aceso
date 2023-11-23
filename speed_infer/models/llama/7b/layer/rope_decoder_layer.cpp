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
#include "layer.h"
#include "models/llama/7b/operation/common_mlp.h"
#include "models/llama/7b/operation/rope_fusion_operation.h"
#include "models/llama/7b/operation/self_attention.h"

namespace atb_speed {
namespace llama_7b {
enum LayerDecoderRopeTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QMIXDWEIGHT,
    IN_KMIXDWEIGHT,
    IN_VMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_SEQLEN,
    OUT_LLAMA7BLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 11;

atb::Status RoperDecoderLayer(const LayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT };
    inputNormNode.outTensorIds = { INTERMIDATE_INPUTNORMOUT };

    atb::infer::LinearParam linearParam = { false, false, false };
    CreateOperation(linearParam, &mixdQLinearNode.operation);
    mixdQLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT };
    mixdQLinearNode.outTensorIds = { INTERMIDATE_MIXEDQ };

    CreateOperation(linearParam, &mixdKLinearNode.operation);
    mixdKLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT };
    mixdKLinearNode.outTensorIds = { INTERMIDATE_MIXEDK };

    CreateOperation(linearParam, &mixdVLinearNode.operation);
    mixdVLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT };
    mixdVLinearNode.outTensorIds = { INTERMIDATE_MIXEDV };

    atb_speed::llama_7b::RopeFusionParam ropeFusionParam;
    ropeFusionParam.headNum = param.headNum;
    atb_speed::llama_7b::RopeFusionOperation(ropeFusionParam, &ropeNode.operation);
    ropeNode.inTensorIds = {
        INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN
    };
    ropeNode.outTensorIds = { INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK };
    
    atb_speed::llama_7b::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.dk = param.dk;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.model = param.model;
    atb_speed::llama_7b::SelfAttentionKvCache(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = { INTERMIDATE_POSITIONEMBEDQ,
        INTERMIDATE_POSITIONEMBEDK,
        INTERMIDATE_MIXEDV,
        IN_ATTENTIONMASK,
        IN_PASTKEY,
        IN_PASTVALUE };
    selfAttentionKvCacheNode.outTensorIds = { INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE };

    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    CreateOperation(linearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = { INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT };
    selfOutLinearNode.outTensorIds = { INTERMIDATE_SELFLINEAROUT };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };

    CreateOperation(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT };
    selfNormNode.outTensorIds = { INTERMIDATE_SELFNORMOUT };

    atb_speed::llama_7b::CommonMlpParam mlpParam;
    atb_speed::llama_7b::CommonMlp(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = { INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT, IN_MLPUPWEIGHT };
    mlpNode.outTensorIds = { INTERMIDATE_MLPOUT };

    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT };
    mlpResidualAddNode.outTensorIds = { OUT_LLAMA7BLAYEROUT };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_PASTKEY);
        const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_PASTVALUE);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(1) = keyTensorDesc;
        outTensorDescs.at(1).shape.dims[1] += 1;
        outTensorDescs.at(2) = valueTensorDesc;
        outTensorDescs.at(2).shape.dims[1] += 1;
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed