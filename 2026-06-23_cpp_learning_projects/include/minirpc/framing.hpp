#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace minirpc {

constexpr std::uint32_t kMaxFrameSize = 4 * 1024 * 1024;
constexpr std::size_t kFrameHeaderSize = 4;

inline std::array<char, kFrameHeaderSize> encode_frame_size(std::uint32_t size) {
    return {
        static_cast<char>((size >> 24) & 0xff),
        static_cast<char>((size >> 16) & 0xff),
        static_cast<char>((size >> 8) & 0xff),
        static_cast<char>(size & 0xff),
    };
}

inline std::uint32_t decode_frame_size(const char* data) {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(data[0])) << 24) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(data[1])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(data[2])) << 8) |
           static_cast<std::uint32_t>(static_cast<unsigned char>(data[3]));
}

inline std::string make_frame(std::string_view payload) {
    auto header = encode_frame_size(static_cast<std::uint32_t>(payload.size()));
    std::string frame;
    frame.reserve(kFrameHeaderSize + payload.size());
    frame.append(header.data(), header.size());
    frame.append(payload.data(), payload.size());
    return frame;
}

enum class FrameReadResult {
    NeedMore,
    Ok,
    TooLarge,
};

inline FrameReadResult try_pop_frame(std::vector<char>& buffer, std::string& payload) {
    if (buffer.size() < kFrameHeaderSize) {
        return FrameReadResult::NeedMore;
    }

    const std::uint32_t size = decode_frame_size(buffer.data());
    if (size > kMaxFrameSize) {
        return FrameReadResult::TooLarge;
    }

    if (buffer.size() < kFrameHeaderSize + size) {
        return FrameReadResult::NeedMore;
    }

    payload.assign(buffer.begin() + static_cast<std::ptrdiff_t>(kFrameHeaderSize),
                   buffer.begin() + static_cast<std::ptrdiff_t>(kFrameHeaderSize + size));
    buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(kFrameHeaderSize + size));
    return FrameReadResult::Ok;
}

}  // namespace minirpc
