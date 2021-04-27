# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from PIL import Image
import numpy as np
import sys
import json
import io
from tritonclient.utils import *
import tritonclient.http as httpclient

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.inception_client = httpclient.InferenceServerClient(url="localhost:8000")
        
    def execute(self, requests):
        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "INPUT")
            preprocess0 = self._preprocess(in0.as_numpy())

            inception_inputs = [
                httpclient.InferInput("input", preprocess0.shape,
                                      np_to_triton_dtype(preprocess0.dtype))
            ]
            inception_inputs[0].set_data_from_numpy(preprocess0)

            inception_outputs = [
                httpclient.InferRequestedOutput("InceptionV3/Predictions/Softmax", class_count=3)
            ]
            
            inception_response = self.inception_client.infer("inception_graphdef",
                                                             inception_inputs,
                                                             outputs = inception_outputs)
            inception_result = inception_response.as_numpy("InceptionV3/Predictions/Softmax")
            
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUTPUT", inception_result)]))

        return responses

    def _preprocess(self, input_np):
        """
        Pre-process an image to meet the inception model size, type and
        format requirements.
        """
        img = Image.open(io.BytesIO(bytearray(input_np)))
        sample_img = img.convert('RGB')
        resized_img = sample_img.resize((299, 299), Image.BILINEAR)
        resized = np.array(resized_img)
        typed = resized.astype(np.float32)
        scaled = (typed / 127.5) - 1

        # Need batch dimension for inception model. Directly setting
        # shape should work in this case since it doesn't require
        # change in data layout.
        nshape = [1]
        nshape.extend(scaled.shape)
        scaled.shape = nshape
        return scaled
