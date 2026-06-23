#include "minirpc/framing.hpp"
#include "minirpc/protobuf_wire.hpp"

#include <cassert>
#include <iostream>

int main() {
    minirpc::RpcMessage request;
    request.id = 42;
    request.kind = minirpc::RpcMessage::Kind::Request;
    request.service = "DemoService";
    request.method = "echo";
    request.payload = "hello";

    const auto encoded = minirpc::ProtobufWireCodec::encode(request);
    const auto decoded = minirpc::ProtobufWireCodec::decode(encoded);
    assert(decoded);
    assert(decoded->id == request.id);
    assert(decoded->kind == request.kind);
    assert(decoded->service == request.service);
    assert(decoded->method == request.method);
    assert(decoded->payload == request.payload);

    auto frame = minirpc::make_frame(encoded);
    std::vector<char> buffer(frame.begin(), frame.end());
    std::string payload;
    const auto result = minirpc::try_pop_frame(buffer, payload);
    assert(result == minirpc::FrameReadResult::Ok);
    assert(payload == encoded);
    assert(buffer.empty());

    std::cout << "protocol tests passed\n";
    return 0;
}

