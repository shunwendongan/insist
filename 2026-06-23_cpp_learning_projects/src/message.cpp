#include "minirpc/message.hpp"

namespace minirpc {

const char* to_string(RpcMessage::Kind kind) {
    switch (kind) {
        case RpcMessage::Kind::Request:
            return "REQUEST";
        case RpcMessage::Kind::Response:
            return "RESPONSE";
        case RpcMessage::Kind::Notify:
            return "NOTIFY";
        case RpcMessage::Kind::Heartbeat:
            return "HEARTBEAT";
        case RpcMessage::Kind::Unknown:
        default:
            return "UNKNOWN";
    }
}

}  // namespace minirpc

