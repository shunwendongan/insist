#pragma once

#include "minirpc/message.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace minirpc {

// A tiny protobuf wire-format codec for proto/rpc.proto.
//
// It intentionally covers only the field types used by RpcMessage:
// varint, length-delimited strings/bytes, and unknown-field skipping.
// The resulting bytes are compatible with the RpcMessage schema.
class ProtobufWireCodec {
public:
    static std::string encode(const RpcMessage& message);
    static std::optional<RpcMessage> decode(std::string_view bytes);

private:
    static void write_varint(std::string& out, std::uint64_t value);
    static void write_tag(std::string& out, std::uint32_t field_number, std::uint32_t wire_type);
    static void write_length_delimited(std::string& out, std::uint32_t field_number, std::string_view value);

    static bool read_varint(std::string_view bytes, std::size_t& pos, std::uint64_t& value);
    static bool skip_field(std::string_view bytes, std::size_t& pos, std::uint32_t wire_type);
};

}  // namespace minirpc

