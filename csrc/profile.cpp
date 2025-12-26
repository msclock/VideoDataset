#include "video_decoder.hpp"

#include <pybind11/embed.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    const auto& video_path =
        "/mnt/public/qiuying/iros/v30/task_2666/videos/observation.images.hand_left/chunk-000/file-000.mp4";
    // const auto& video_path = "/home/zy/Downloads/file-000.mp4";
    pybind11::scoped_interpreter guard{};
    auto decoder = VideoDecoder(0, "h265");
    for (int i = 0; i < 40000; i++)
        auto np_frame = decoder.decodeToTensor(video_path, i);
    return 0;
}
