#ifndef __ANN_SERIALIZER_H__
#define __ANN_SERIALIZER_H__

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <string>

#include "Cluster.h"
#include "DataTypes.h"

class Serializer {
  public:
    template <typename T>
    static void ToFile(T data, const std::string &file_name) {
        std::ofstream os(file_name, std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(os);
        archive(data);
    }

    template <typename T> static T FromFile(const std::string &file_name) {
        T data;
        std::ifstream ifs(file_name, std::ios::binary);
        cereal::PortableBinaryInputArchive iarchive(
            ifs); // Create an input archive
        iarchive(data);
        return data;
    }
};

#endif