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
#ifndef ATB_SPEED_MODELS_LLAMA_7B_ROPE_MODEL_H
#define ATB_SPEED_MODELS_LLAMA_7B_ROPE_MODEL_H

#include "atb_speed/base/model.h"

namespace atb_speed {
namespace llama_7b {
class DecoderRopeModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;

        void FromString(const std::string &param);
    };

    explicit DecoderRopeModel(const std::string &param);

    ~DecoderRopeModel();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;

    Param param_;
};
class EncoderRopeModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;

        void FromString(const std::string &param);
    };

    explicit EncoderRopeModel(const std::string &param);

    ~EncoderRopeModel();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;

    Param param_;
};

class EncoderRopeEmbModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;

        void FromString(const std::string &param);
    };

    explicit EncoderRopeEmbModel(const std::string &param);

    ~EncoderRopeEmbModel();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;

    Param param_;
};

} // namespace llama_7b
} // namespace atb_speed
#endif