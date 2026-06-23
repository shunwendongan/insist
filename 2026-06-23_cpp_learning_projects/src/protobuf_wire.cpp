#include "minirpc/protobuf_wire.hpp"

#include <limits>
#include <utility>

namespace minirpc {

namespace {

constexpr std::uint32_t kWireVarint = 0;
constexpr std::uint32_t kWireFixed64 = 1;
constexpr std::uint32_t kWireLengthDelimited = 2;
constexpr std::uint32_t kWireFixed32 = 5;

}  // namespace

std::string ProtobufWireCodec::encode(const RpcMessage& message) {
    std::string out;

    if (message.id != 0) {
        write_tag(out, 1, kWireVarint);
        write_varint(out, message.id);
    }

    if (message.kind != RpcMessage::Kind::Unknown) {
        write_tag(out, 2, kWireVarint);
        write_varint(out, static_cast<std::uint32_t>(message.kind));
    }

    if (!message.service.empty()) {
        write_length_delimited(out, 3, message.service);
    }

    if (!message.method.empty()) {
        write_length_delimited(out, 4, message.method);
    }

    if (!message.payload.empty()) {
        write_length_delimited(out, 5, message.payload);
    }

    if (message.status != 0) {
        write_tag(out, 6, kWireVarint);
        write_varint(out, static_cast<std::uint32_t>(message.status));
    }

    if (!message.error.empty()) {
        write_length_delimited(out, 7, message.error);
    }

    return out;
}

std::optional<RpcMessage> ProtobufWireCodec::decode(std::string_view bytes) {
    RpcMessage message;
    std::size_t pos = 0;

    while (pos < bytes.size()) {
        std::uint64_t tag = 0;
        if (!read_varint(bytes, pos, tag)) {
            return std::nullopt;
        }

        const auto field_number = static_cast<std::uint32_t>(tag >> 3);
        const auto wire_type = static_cast<std::uint32_t>(tag & 0x07);

        switch (field_number) {
            case 1: {
                if (wire_type != kWireVarint) {
                    return std::nullopt;
                }
                std::uint64_t value = 0;
                if (!read_varint(bytes, pos, value)) {
                    return std::nullopt;
                }
                message.id = value;
                break;
            }
            case 2: {
                if (wire_type != kWireVarint) {
                    return std::nullopt;
                }
                std::uint64_t value = 0;
                if (!read_varint(bytes, pos, value)) {
                    return std::nullopt;
                }
                message.kind = static_cast<RpcMessage::Kind>(value);
                break;
            }
            case 3:
            case 4:
            case 5:
            case 7: {
                if (wire_type != kWireLengthDelimited) {
                    return std::nullopt;
                }
                std::uint64_t size = 0;
                if (!read_varint(bytes, pos, size)) {
                    return std::nullopt;
                }
                if (size > bytes.size() - pos) {
                    return std::nullopt;
                }
                std::string value(bytes.substr(pos, static_cast<std::size_t>(size)));
                pos += static_cast<std::size_t>(size);

                if (field_number == 3) {
                    message.service = std::move(value);
                } else if (field_number == 4) {
                    message.method = std::move(value);
                } else if (field_number == 5) {
                    message.payload = std::move(value);
                } else {
                    message.error = std::move(value);
                }
                break;
            }
            case 6: {
                if (wire_type != kWireVarint) {
                    return std::nullopt;
                }
                std::uint64_t value = 0;
                if (!read_varint(bytes, pos, value)) {
                    return std::nullopt;
                }
                if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                    return std::nullopt;
                }
                message.status = static_cast<std::int32_t>(value);
                break;
            }
            default:
                if (!skip_field(bytes, pos, wire_type)) {
                    return std::nullopt;
                }
                break;
        }
    }

    return message;
}

void ProtobufWireCodec::write_varint(std::string& out, std::uint64_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<char>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void ProtobufWireCodec::write_tag(std::string& out, std::uint32_t field_number, std::uint32_t wire_type) {
    write_varint(out, (static_cast<std::uint64_t>(field_number) << 3) | wire_type);
}

void ProtobufWireCodec::write_length_delimited(std::string& out,
                                               std::uint32_t field_number,
                                               std::string_view value) {
    write_tag(out, field_number, kWireLengthDelimited);
    write_varint(out, value.size());
    out.append(value.data(), value.size());
}

bool ProtobufWireCodec::read_varint(std::string_view bytes, std::size_t& pos, std::uint64_t& value) {
    value = 0;
    int shift = 0;

    while (pos < bytes.size() && shift <= 63) {
        const auto byte = static_cast<unsigned char>(bytes[pos++]);
        value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) {
            return true;
        }
        shift += 7;
    }

    return false;
}

bool ProtobufWireCodec::skip_field(std::string_view bytes, std::size_t& pos, std::uint32_t wire_type) {
    switch (wire_type) {
        case kWireVarint: {
            std::uint64_t ignored = 0;
            return read_varint(bytes, pos, ignored);
        }
        case kWireFixed64:
            if (bytes.size() - pos < 8) {
                return false;
            }
            pos += 8;
            return true;
        case kWireLengthDelimited: {
            std::uint64_t size = 0;
            if (!read_varint(bytes, pos, size) || size > bytes.size() - pos) {
                return false;
            }
            pos += static_cast<std::size_t>(size);
            return true;
        }
        case kWireFixed32:
            if (bytes.size() - pos < 4) {
                return false;
            }
            pos += 4;
            return true;
        default:
            return false;
    }
}

}  // namespace minirpc
