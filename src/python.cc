// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "pb_env.h"
#include "pb_utils.h"
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"


#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return nullptr;                                                  \
    }                                                                  \
  } while (false)

#define RESPOND_ALL_AND_RETURN_IF_EXCEPTION(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                     \
    try {                                                                  \
      (X);                                                                 \
    }                                                                      \
    catch (const PythonBackendException& exception) {                      \
      TRITONSERVER_Error* raarie_err__ = TRITONSERVER_ErrorNew(            \
          TRITONSERVER_ERROR_INTERNAL, exception.what());                  \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__);     \
      return nullptr;                                                      \
    }                                                                      \
  } while (false)

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      return rarie_err__;                                               \
    }                                                                   \
  } while (false)

#define RESPOND_AND_RETURN_IF_EXCEPTION(REQUEST, X)                     \
  do {                                                                  \
    try {                                                               \
      (X);                                                              \
    }                                                                   \
    catch (const PythonBackendException& exception) {                   \
      TRITONSERVER_Error* rarie_err__ = TRITONSERVER_ErrorNew(          \
          TRITONSERVER_ERROR_INTERNAL, exception.what());               \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      return rarie_err__;                                               \
    }                                                                   \
  } while (false)

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

#define GUARDED_RESPOND_IF_EXCEPTION(RESPONSES, IDX, X)                 \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      try {                                                             \
        (X);                                                            \
      }                                                                 \
      catch (const PythonBackendException& pb_exception) {              \
        TRITONSERVER_Error* err__ = TRITONSERVER_ErrorNew(              \
            TRITONSERVER_ERROR_INTERNAL, pb_exception.what());          \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

#define RETURN_IF_EXCEPTION(X)                                 \
  do {                                                         \
    try {                                                      \
      (X);                                                     \
    }                                                          \
    catch (const PythonBackendException& pb_exception) {       \
      TRITONSERVER_Error* rarie_err__ = TRITONSERVER_ErrorNew( \
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());   \
      return rarie_err__;                                      \
    }                                                          \
  } while (false)

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

struct BackendState {
  std::string python_lib;
  int64_t shm_default_byte_size;
  int64_t shm_growth_byte_size;
  int64_t stub_timeout_seconds;
  std::unique_ptr<EnvironmentManager> env_manager;
};

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get backend state
  BackendState* StateForBackend() { return backend_state_; }

  // Get the Python execution environment
  std::string PythonExecutionEnv() { return python_execution_env_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  BackendState* backend_state_;
  std::string python_execution_env_;
};

TRITONSERVER_Error*
CreateTritonErrorFromException(const PythonBackendException& pb_exception)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
}

class ModelInstanceState : public BackendModelInstance {
  ModelInstanceState(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance);

  TRITONBACKEND_Model* triton_model_;
  bi::interprocess_mutex* stub_mutex_;
  bi::interprocess_condition* stub_cond_;
  bi::interprocess_mutex* parent_mutex_;
  bi::interprocess_condition* parent_cond_;
  bi::interprocess_mutex* health_mutex_;
  std::unique_ptr<bi::scoped_lock<bi::interprocess_mutex>> parent_lock_;
  std::string model_path_;
  IPCMessage* ipc_message_;
  std::unique_ptr<SharedMemory> shm_pool_;

  // Stub process pid
  pid_t stub_pid_;

  // Parent process pid
  pid_t parent_pid_;
  bool initialized_;

  // Path to python execution environment
  std::string path_to_libpython_;
  std::string path_to_activate_;

 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      ModelInstanceState** model_instance_state);

  ~ModelInstanceState();

  // Load Triton inputs to the appropriate Protobufs
  TRITONSERVER_Error* GetInputTensor(
      const uint32_t input_idx, Tensor* input_tensor,
      TRITONBACKEND_Request* request,
      std::vector<TRITONBACKEND_Response*>& responses);

  TRITONSERVER_Error* ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Create the stub process.
  TRITONSERVER_Error* SetupStubProcess();

  // Notifies the stub process on the new request.  Returns false if the parent
  // process fails to acquire the lock.
  bool NotifyStub();

  // Checks whether the stub process is live
  bool IsStubProcessAlive();

  // Wait for stub notification
  bool WaitForStubNotification();

  // Responds to all the requests with an error message.
  void RespondErrorToAllRequests(
      const char* message, std::vector<TRITONBACKEND_Response*>& responses,
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Kill stub process
  void KillStubProcess();

  // Start stub process
  TRITONSERVER_Error* StartStubProcess();
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance), stub_pid_(0),
      initialized_(false)
{
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;  // success
}

bool
ModelInstanceState::NotifyStub()
{
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::milliseconds(1000);
  bi::scoped_lock<bi::interprocess_mutex> lock(*stub_mutex_, timeout);

  if (lock) {
    stub_cond_->notify_one();
    return true;
  } else {
    return false;
  }
}

void
ModelInstanceState::KillStubProcess()
{
  kill(stub_pid_, SIGKILL);
  int status;
  waitpid(stub_pid_, &status, 0);
  stub_pid_ = 0;
}

bool
ModelInstanceState::WaitForStubNotification()
{
  uint64_t timeout_seceonds = 1000;
  boost::posix_time::ptime timeout =
      boost::get_system_time() +
      boost::posix_time::milliseconds(timeout_seceonds);

  {
    bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

    // Check if lock has been acquired.
    if (lock) {
      ipc_message_->health = false;
    } else {
      // If It failed to obtain the lock, it means that the stub has been
      // stuck or exited while holding the health mutex lock.
      return false;
    }
  }

  timeout = boost::get_system_time() +
            boost::posix_time::milliseconds(timeout_seceonds);
  while (!parent_cond_->timed_wait(*parent_lock_, timeout)) {
    if (!IsStubProcessAlive()) {
      return false;
    }

    timeout = boost::get_system_time() +
              boost::posix_time::milliseconds(timeout_seceonds);
  }
  return true;
}

void
ModelInstanceState::RespondErrorToAllRequests(
    const char* message, std::vector<TRITONBACKEND_Response*>& responses,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    if (responses[r] == nullptr)
      continue;

    TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("Failed to process the request(s), message: ") + message)
            .c_str());
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, "Failed to process the batch of requests.");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
        "failed sending response");

    responses[r] = nullptr;
    TRITONSERVER_ErrorDelete(err);
  }
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int max_batch_size = model_state->MaxBatchSize();
  std::string name = model_state->Name();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.

  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Python backend for '" + name + "'")
                  .c_str()));
      return nullptr;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return nullptr;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return nullptr;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                name + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return nullptr;
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " + Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Create Python inference requests
  RequestBatch* request_batch;
  off_t request_batch_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&request_batch, sizeof(RequestBatch), request_batch_offset));

  ipc_message_->request_batch = request_batch_offset;
  request_batch->batch_size = request_count;

  Request* requests_shm;
  off_t requests_shm_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&requests_shm, sizeof(Request) * request_count,
      requests_shm_offset));
  request_batch->requests = requests_shm_offset;

  // We take the responsibilty of the responses.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response.");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    Request* python_infer_request = &requests_shm[r];
    uint32_t requested_input_count = 0;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count,
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    python_infer_request->requested_input_count = requested_input_count;

    uint32_t requested_output_count = 0;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
    python_infer_request->requested_output_count = requested_output_count;

    Tensor* input_tensors;
    off_t input_tensors_offset;

    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        &responses, request_count,
        shm_pool_->Map(
            (char**)&input_tensors, sizeof(Tensor) * requested_input_count,
            input_tensors_offset));
    python_infer_request->inputs = input_tensors_offset;

    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      Tensor* input_tensor = &input_tensors[iidx];

      RESPOND_ALL_AND_RETURN_IF_ERROR(
          &responses, request_count,
          GetInputTensor(iidx, input_tensor, request, responses));
    }

    off_t* requested_output_names;
    off_t requested_output_names_offset;

    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        &responses, request_count,
        shm_pool_->Map(
            (char**)&requested_output_names,
            sizeof(off_t) * requested_output_count,
            requested_output_names_offset));
    python_infer_request->requested_output_names =
        requested_output_names_offset;

    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          &responses, request_count,
          TRITONBACKEND_RequestOutputName(
              request, iidx, &requested_output_name));

      // output name
      off_t output_name_offset;
      RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
          &responses, request_count,
          SaveStringToSharedMemory(
              shm_pool_, output_name_offset, requested_output_name));
      requested_output_names[iidx] = output_name_offset;
    }

    // request id
    const char* id;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count, TRITONBACKEND_RequestId(request, &id));

    off_t id_offset;
    RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
        &responses, request_count,
        SaveStringToSharedMemory(shm_pool_, id_offset, id));
    python_infer_request->id = id_offset;

    uint64_t correlation_id;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        &responses, request_count,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));
    python_infer_request->correlation_id = correlation_id;
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // This means that the stub process has exited and Python
  // backend failed to restart the stub process.
  if (stub_pid_ == 0) {
    const char* error_message = "The stub process has exited unexpectedly.";
    RespondErrorToAllRequests(
        error_message, responses, requests, request_count);

    // Update the shared memory offset so that we can reuse the shared memory
    shm_pool_->SetOffset(request_batch_offset);
    return nullptr;
  }

  // If parent fails to notify the stub or the stub fails to notify the
  // parent in a timely manner, kill the stub process and restart the
  // stub process.
  if (!NotifyStub() || !WaitForStubNotification()) {
    KillStubProcess();
    const char* error_message = "The stub process has exited unexpectedly.";
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error_message);
    TRITONSERVER_Error* err = StartStubProcess();
    if (err == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO, "Stub process successfully restarted.");
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string(
               "Stub process failed to restart. Your future requests to "
               "model ") +
           name_ + " will fail. Error: " + TRITONSERVER_ErrorMessage(err))
              .c_str());
    }
    RespondErrorToAllRequests(
        error_message, responses, requests, request_count);

    // Update the shared memory offset so that we can reuse the shared memory
    shm_pool_->SetOffset(request_batch_offset);
    return nullptr;
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Parsing the request response
  ResponseBatch* response_batch;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      &responses, request_count,
      shm_pool_->MapOffset(
          (char**)&response_batch, sizeof(ResponseBatch),
          ipc_message_->response_batch));

  // If inference fails, release all the requests and send an error response. If
  // inference fails at this stage, it usually indicates a bug in the model code
  if (response_batch->has_error) {
    if (response_batch->is_error_set) {
      char* error_message;
      RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
          &responses, request_count,
          LoadStringFromSharedMemory(
              shm_pool_, response_batch->error, error_message));
      RespondErrorToAllRequests(
          error_message, responses, requests, request_count);
    } else {
      const char* error_message =
          "Failed to fetch the error in response batch.";
      RespondErrorToAllRequests(
          error_message, responses, requests, request_count);
    }

    return nullptr;
  }

  Response* responses_shm;
  RESPOND_ALL_AND_RETURN_IF_EXCEPTION(
      &responses, request_count,
      shm_pool_->MapOffset(
          (char**)&responses_shm, sizeof(Response) * response_batch->batch_size,
          response_batch->responses));


  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = responses[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;

    // Get response r
    Response* response_shm = &responses_shm[r];

    if (response_shm->has_error) {
      try {
        if (response_shm->is_error_set) {
          char* err_string;
          LoadStringFromSharedMemory(
              shm_pool_, response_shm->error, err_string);
          TRITONSERVER_Error* err =
              TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_string);

          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed sending response");
          TRITONSERVER_ErrorDelete(err);
        } else {
          const char* err_string = "Failed to process response.";
          TRITONSERVER_Error* err =
              TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_string);

          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
              "failed sending response");
          TRITONSERVER_ErrorDelete(err);
        }
      }
      catch (const PythonBackendException& pb_exception) {
        TRITONSERVER_Error* err = CreateTritonErrorFromException(pb_exception);

        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
            "failed sending response");
      }

      responses[r] = nullptr;

      // If has_error is true, we do not look at the response even if the
      // response is set.
      continue;
    }

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    Tensor* output_tensors;
    GUARDED_RESPOND_IF_EXCEPTION(
        responses, r,
        shm_pool_->MapOffset(
            (char**)&output_tensors, sizeof(Tensor) * requested_output_count,
            response_shm->outputs));

    bool cuda_copy = false;
    std::set<std::string> requested_output_names;
    for (size_t j = 0; j < requested_output_count; ++j) {
      const char* output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, j, &output_name));
      requested_output_names.insert(output_name);
    }

    for (size_t j = 0; j < requested_output_count; ++j) {
      Tensor* output_tensor = &output_tensors[j];
      TRITONSERVER_DataType triton_dt = output_tensor->dtype;
      size_t dims_count = output_tensor->dims_count;
      int64_t* dims;
      GUARDED_RESPOND_IF_EXCEPTION(
          responses, r,
          shm_pool_->MapOffset(
              (char**)&dims, sizeof(int64_t) * dims_count,
              output_tensor->dims));

      char* name;
      GUARDED_RESPOND_IF_EXCEPTION(
          responses, r,
          LoadStringFromSharedMemory(shm_pool_, output_tensor->name, name));

      // Skip the output tensor if it is not in the list of requested outputs
      if (requested_output_names.find(std::string(name)) ==
          requested_output_names.end()) {
        continue;
      }

      RawData* raw_data;
      GUARDED_RESPOND_IF_EXCEPTION(
          responses, r,
          shm_pool_->MapOffset(
              (char**)&raw_data, sizeof(RawData), output_tensor->raw_data));

      char* data;
      GUARDED_RESPOND_IF_EXCEPTION(
          responses, r,
          shm_pool_->MapOffset(
              (char**)&data, raw_data->byte_size, raw_data->memory_ptr));

      std::vector<int64_t> batch_shape(dims, dims + dims_count);
      TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t actual_memory_type_id = 0;
      void* buffer;

      TRITONBACKEND_Output* response_output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &response_output, name, triton_dt, batch_shape.data(),
              batch_shape.size()));

      bool cuda_used;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              response_output, &buffer, raw_data->byte_size,
              &actual_memory_type, &actual_memory_type_id));
      CopyBuffer(
          "Failed to copy string", TRITONSERVER_MEMORY_CPU /* memory_type */,
          0 /* memory_type_id */, actual_memory_type, actual_memory_type_id,
          raw_data->byte_size, data, buffer, CudaStream(), &cuda_used);
      cuda_copy |= cuda_used;
    }
#ifdef TRITON_ENABLE_GPU
    if (cuda_copy) {
      cudaStreamSynchronize(stream_);
    }
#endif  // TRITON_ENABLE_GPU

    // If error happens at this stage, we can only log it
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
        "failed sending response");
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       Name() + " released " + std::to_string(request_count) + " requests")
          .c_str());

  // Update the shared memory offset so that we can reuse the shared memory
  shm_pool_->SetOffset(request_batch_offset);
  return nullptr;
}

bool
ModelInstanceState::IsStubProcessAlive()
{
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::seconds(1);
  bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_, timeout);

  // Check if lock has been acquired.
  if (lock) {
    return ipc_message_->health;
  } else {
    // If It failed to obtain the lock, it means that the stub has been
    // stuck or exited while holding the health mutex lock.
    return false;
  }
}

TRITONSERVER_Error*
ModelInstanceState::StartStubProcess()
{
  stub_mutex_ = new (stub_mutex_) bi::interprocess_mutex;
  health_mutex_ = new (health_mutex_) bi::interprocess_mutex;
  stub_cond_ = new (stub_cond_) bi::interprocess_condition;

  std::string kind = TRITONSERVER_InstanceGroupKindString(kind_);
  std::string shm_region_name =
      std::string("/") + Name() + "_" + kind + "_" + std::to_string(device_id_);

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int64_t shm_growth_size =
      model_state->StateForBackend()->shm_growth_byte_size;
  int64_t shm_default_size =
      model_state->StateForBackend()->shm_default_byte_size;
  const char* model_path = model_state->RepositoryPath().c_str();

  initialized_ = false;

  pid_t pid = fork();
  if (pid < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Failed to fork the stub process.");
  }

  // Stub process
  if (pid == 0) {
    const char* stub_args[4];
    stub_args[0] = "bash";
    stub_args[1] = "-c";
    stub_args[3] = nullptr;  // Last argument must be nullptr

    // Default Python backend stub
    std::string python_backend_stub =
        model_state->StateForBackend()->python_lib +
        "/triton_python_backend_stub";

    // Path to alternative Python backend stub
    std::string model_python_backend_stub =
        std::string(model_path) + "/triton_python_backend_stub";

    if (FileExists(model_python_backend_stub)) {
      python_backend_stub = model_python_backend_stub;
    }

    std::stringstream ss;
    ss << "exec " << python_backend_stub << " " << model_path_ << " "
       << shm_region_name << " " << shm_default_size << " " << shm_growth_size
       << " " << parent_pid_ << " "
       << model_state->StateForBackend()->python_lib;

    std::string bash_argument;
    bash_argument = ss.str();
    if (model_state->PythonExecutionEnv() != "") {
      // Need to properly set the LD_LIBRARY_PATH so that Python environments
      // using different python versions load properly.
      bash_argument = "export LD_LIBRARY_PATH=" + path_to_libpython_ +
                      ":$LD_LIBRARY_PATH; source " + path_to_activate_ +
                      " && " + bash_argument;
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Starting Python backend stub: ") + bash_argument)
            .c_str());

    stub_args[2] = bash_argument.c_str();
    if (execvp("bash", (char**)stub_args) == -1) {
      std::stringstream ss;
      ss << "Failed to run python backend stub. Errno = " << errno << '\n'
         << "Python backend stub path: " << python_backend_stub << '\n'
         << "Shared Memory Region Name: " << shm_region_name << '\n'
         << "Shared Memory Default Byte Size: " << shm_default_size << '\n'
         << "Shared Memory Growth Byte Size: " << shm_growth_size << '\n';
      std::string log_message = ss.str();
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, log_message.c_str());

      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to initialize model instance ") + Name())
              .c_str());
    }

  } else {
    int64_t stub_timeout_seconds =
        model_state->StateForBackend()->stub_timeout_seconds;

    stub_pid_ = pid;
    boost::posix_time::ptime timeout =
        boost::get_system_time() +
        boost::posix_time::seconds(stub_timeout_seconds);

    // Pre initialization step.
    if (!parent_cond_->timed_wait(*parent_lock_, timeout)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Timed out occurred while waiting for the stub process. "
                       "Failed to initialize model instance ") +
           Name())
              .c_str());
    }

    triton::common::TritonJson::WriteBuffer buffer;
    Model()->ModelConfig().Write(&buffer);

    std::unordered_map<std::string, std::string> initialize_args = {
        {"model_config", buffer.MutableContents()},
        {"model_instance_kind", TRITONSERVER_InstanceGroupKindString(kind_)},
        {"model_instance_name", name_},
        {"model_instance_device_id", std::to_string(device_id_)},
        {"model_repository", model_state->RepositoryPath()},
        {"model_version", std::to_string(model_state->Version())},
        {"model_name", model_state->Name()}};

    off_t initialize_args_offset;
    RETURN_IF_EXCEPTION(SaveMapToSharedMemory(
        shm_pool_, initialize_args_offset, initialize_args));
    ipc_message_->request_batch = initialize_args_offset;

    // If parent fails to notify the stub or the stub fails to notify the
    // parent in a timely manner, kill the stub process and restart the
    // stub process.
    if (!NotifyStub() || !WaitForStubNotification()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to initialize stub, stub process exited "
                       "unexpectedly: ") +
           name_)
              .c_str());
    }

    ResponseBatch* response_batch;
    RETURN_IF_EXCEPTION(shm_pool_->MapOffset(
        (char**)&response_batch, sizeof(RequestBatch),
        ipc_message_->response_batch));

    if (response_batch->has_error) {
      char* err_message;
      RETURN_IF_EXCEPTION(LoadStringFromSharedMemory(
          shm_pool_, response_batch->error, err_message));
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_message);
    }

    initialized_ = true;
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::SetupStubProcess()
{
  std::string kind = TRITONSERVER_InstanceGroupKindString(kind_);
  std::string shm_region_name =
      std::string("/") + Name() + "_" + kind + "_" + std::to_string(device_id_);

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());
  int64_t shm_growth_size =
      model_state->StateForBackend()->shm_growth_byte_size;
  int64_t shm_default_size =
      model_state->StateForBackend()->shm_default_byte_size;

  try {
    shm_pool_ = std::make_unique<SharedMemory>(
        shm_region_name, shm_default_size, shm_growth_size,
        true /* truncate */);
  }
  catch (const PythonBackendException& pb_exception) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
  }

  // Stub mutex and CV
  bi::interprocess_mutex* stub_mutex;
  off_t stub_mutex_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&stub_mutex, sizeof(bi::interprocess_mutex), stub_mutex_offset));
  stub_mutex = new (stub_mutex) bi::interprocess_mutex;

  bi::interprocess_condition* stub_cv;
  off_t stub_cv_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&stub_cv, sizeof(bi::interprocess_condition), stub_cv_offset));
  stub_cv = new (stub_cv) bi::interprocess_condition;

  stub_cond_ = stub_cv;
  stub_mutex_ = stub_mutex;

  // Parent Mutex and CV
  bi::interprocess_mutex* parent_mutex;
  off_t parent_mutex_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&parent_mutex, sizeof(bi::interprocess_mutex),
      parent_mutex_offset));
  parent_mutex = new (parent_mutex) bi::interprocess_mutex;

  bi::interprocess_condition* parent_cv;
  off_t parent_cv_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&parent_cv, sizeof(bi::interprocess_condition),
      parent_cv_offset));
  parent_cv = new (parent_cv) bi::interprocess_condition;

  bi::interprocess_mutex* health_mutex;
  off_t health_mutex_offset;
  RETURN_IF_EXCEPTION(shm_pool_->Map(
      (char**)&health_mutex, sizeof(bi::interprocess_mutex),
      health_mutex_offset));
  health_mutex = new (health_mutex) bi::interprocess_mutex;

  parent_cond_ = parent_cv;
  parent_mutex_ = parent_mutex;
  health_mutex_ = health_mutex;
  parent_lock_ =
      std::make_unique<bi::scoped_lock<bi::interprocess_mutex>>(*parent_mutex);

  off_t ipc_offset;
  RETURN_IF_EXCEPTION(
      shm_pool_->Map((char**)&ipc_message_, sizeof(IPCMessage), ipc_offset));

  uint64_t model_version = model_state->Version();
  const char* model_path = model_state->RepositoryPath().c_str();

  std::stringstream ss;
  // Use <path>/version/model.py as the model location
  ss << model_path << "/" << model_version << "/model.py";
  model_path_ = ss.str();
  struct stat buffer;

  // Check if model.py exists
  if (stat(model_path_.c_str(), &buffer) != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("model.py does not exist in the model repository path: " + model_path_)
            .c_str());
  }

  // Path to the extracted Python env
  std::string python_execution_env = "";
  if (model_state->PythonExecutionEnv() != "") {
    try {
      python_execution_env =
          model_state->StateForBackend()->env_manager->ExtractIfNotExtracted(
              model_state->PythonExecutionEnv());
    }
    catch (PythonBackendException& pb_exception) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, pb_exception.what());
    }

    path_to_activate_ = python_execution_env + "/bin/activate";
    path_to_libpython_ = python_execution_env + "/lib";
    if (python_execution_env.length() > 0 && !FileExists(path_to_activate_)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Path ") + path_to_activate_ +
           " does not exist. The Python environment should contain an "
           "'activate' script.")
              .c_str());
    }
  }

  parent_pid_ = getpid();
  RETURN_IF_ERROR(StartStubProcess());

  return nullptr;
}

ModelInstanceState::~ModelInstanceState()
{
  if (initialized_) {
    {
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
      ipc_message_->health = false;
    }

    // Sleep 1 second so that the child process has a chance to change the
    // health variable
    sleep(1);

    bool healthy = false;
    {
      bi::scoped_lock<bi::interprocess_mutex> lock(*health_mutex_);
      healthy = ipc_message_->health;
    }

    if (healthy) {
      // Signal to the termination to the Python backend stub using a request of
      // size 0.
      RequestBatch* request_batch;
      off_t request_batch_offset;
      shm_pool_->Map(
          (char**)&request_batch, sizeof(RequestBatch), request_batch_offset);
      request_batch->batch_size = 0;
      ipc_message_->request_batch = request_batch_offset;

      if (NotifyStub()) {
        // Wait for stub notification
        parent_cond_->wait(*parent_lock_);
      }
    }
  }

  // Terminate the stub process if it has been created.
  if (stub_pid_ != 0) {
    int status;
    kill(stub_pid_, SIGTERM);
    waitpid(stub_pid_, &status, 0);
  }

  // Destory the lock before deletion of shared memory is triggered.
  parent_lock_.reset(nullptr);
}

TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, Tensor* input_tensor,
    TRITONBACKEND_Request* request,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  const char* input_name;
  // Load iidx'th input name
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInputName(request, input_idx, &input_name));

  // Load iidx'th input
  TRITONBACKEND_Input* in;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, input_name, &in));

  // Load input properties
  TRITONSERVER_DataType input_dtype;
  const int64_t* input_shape;
  uint32_t input_dims_count;
  uint64_t input_byte_size;
  uint32_t input_buffer_count;

  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      in, &input_name, &input_dtype, &input_shape, &input_dims_count,
      &input_byte_size, &input_buffer_count));

  // If input_byte_size is larger than 2GBs, reject request the request.
  uint64_t max_input_size = INT32_MAX;
  if (input_byte_size > max_input_size) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Python backend does not support input size larger than 2GBs, consider "
        "partitioning your input into multiple inputs.");
  }

  // We need to create a new collector for every request because python backend
  // sends each request individually to the python model
  BackendInputCollector collector(
      &request, 1, &responses, Model()->TritonMemoryManager(),
      false /* pinned_enable */, CudaStream());

  const TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  const int memory_type_id = 0;

  char* input_buffer;
  RETURN_IF_EXCEPTION(SaveTensorToSharedMemory(
      shm_pool_, input_tensor, input_buffer, memory_type, memory_type_id,
      input_byte_size, input_name, input_shape, input_dims_count, input_dtype));

  // Load raw data into input_tensor raw data.
  // FIXME: Avoid the copy to CPU Memory when
  // the data is in GPU.
  collector.ProcessTensor(
      input_name, input_buffer, input_byte_size, memory_type, memory_type_id);

  return nullptr;
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));

  const char* path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));
  python_execution_env_ = "";

  void* bstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &bstate));
  backend_state_ = reinterpret_cast<BackendState*>(bstate);
  triton::common::TritonJson::Value params;
  if (model_config_.Find("parameters", &params)) {
    // Skip the EXECUTION_ENV_PATH variable if it doesn't exist.
    TRITONSERVER_Error* error =
        GetParameterValue(params, "EXECUTION_ENV_PATH", &python_execution_env_);
    if (error == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Using Python execution env ") + python_execution_env_)
              .c_str());
    } else {
      // Delete the error
      TRITONSERVER_ErrorDelete(error);
    }
  }

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw triton::backend::BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported artifact type for model '") + Name() + "'")
            .c_str()));
  }
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  // Check backend version to ensure compatibility
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendState> backend_state(new BackendState());
  triton::common::TritonJson::Value cmdline;
  backend_state->shm_default_byte_size = 64 * 1024 * 1024;  // 64 MBs
  backend_state->shm_growth_byte_size = 64 * 1024 * 1024;   // 64 MBs
  backend_state->stub_timeout_seconds = 30;

  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value shm_growth_size;
    std::string shm_growth_byte_size;
    if (cmdline.Find("shm-growth-byte-size", &shm_growth_size)) {
      RETURN_IF_ERROR(shm_growth_size.AsString(&shm_growth_byte_size));
      try {
        backend_state->shm_growth_byte_size = std::stol(shm_growth_byte_size);
        if (backend_state->shm_growth_byte_size <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-growth-byte-size") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value shm_default_size;
    std::string shm_default_byte_size;
    if (cmdline.Find("shm-default-byte-size", &shm_default_size)) {
      RETURN_IF_ERROR(shm_default_size.AsString(&shm_default_byte_size));
      try {
        backend_state->shm_default_byte_size = std::stol(shm_default_byte_size);
        // Shared memory default byte size can't be less than 4 MBs.
        if (backend_state->shm_default_byte_size < 4 * 1024 * 1024) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shm-default-byte-size") +
               " can't be smaller than 4 MiBs")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }

    triton::common::TritonJson::Value stub_timeout_seconds;
    std::string stub_timeout_string_seconds;
    if (cmdline.Find("stub-timeout-seconds", &stub_timeout_seconds)) {
      RETURN_IF_ERROR(
          stub_timeout_seconds.AsString(&stub_timeout_string_seconds));
      try {
        backend_state->stub_timeout_seconds =
            std::stol(stub_timeout_string_seconds);
        if (backend_state->stub_timeout_seconds <= 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("stub-timeout-seconds") +
               " can't be smaller than or equal to zero.")
                  .c_str());
        }
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
      }
    }
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("shm-default-byte-size=") +
       std::to_string(backend_state->shm_default_byte_size) +
       ",shm-growth-byte-size=" +
       std::to_string(backend_state->shm_growth_byte_size) +
       ",stub-timeout-seconds=" +
       std::to_string(backend_state->stub_timeout_seconds))
          .c_str());

  // Use BackendArtifacts to determine the location of Python files
  const char* location;
  TRITONBACKEND_ArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &location));
  backend_state->python_lib = location;
  backend_state->env_manager = std::make_unique<EnvironmentManager>();

  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backend_state.get())));

  backend_state.release();
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: Start");
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto backend_state = reinterpret_cast<BackendState*>(vstate);
  delete backend_state;
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: End");
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  RETURN_IF_ERROR(instance_state->SetupStubProcess());
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  RETURN_IF_ERROR(instance_state->ProcessRequests(requests, request_count));

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::python
