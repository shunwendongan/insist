#pragma once

#include <cstdint>
#include <string>

namespace minirpc {

struct RpcMessage {
    enum class Kind : std::uint32_t {
        Unknown = 0,
        Request = 1,
        Response = 2,
        Notify = 3,
        Heartbeat = 4,
    };

    std::uint64_t id = 0;
    Kind kind = Kind::Unknown;
    std::string service;
    std::string method;
    std::string payload;
    std::int32_t status = 0;
    std::string error;
};

const char* to_string(RpcMessage::Kind kind);

}  // namespace minirpc

