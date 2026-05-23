
#include "CsrReader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

#include "Timer.h"
#include "TypeConvert.h"
#include "common.h"

CsrMetaData CsrReader::Read(const std::string file_path, const int batch_size,
                            sparse_vector_processor callback,
                            set_metadata set_meta) {
    Timer t("Load file: " + file_path);
    std::ifstream file_data(file_path, std::ios::binary);
    std::ifstream file_index(file_path, std::ios::binary);
    CsrMetaData metaData{};
    if (!file_data.is_open() || !file_index.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return metaData;
    }

    file_index.read(reinterpret_cast<char *>(&metaData.n_row),
                    sizeof(metadata_t));
    file_index.read(reinterpret_cast<char *>(&metaData.n_col),
                    sizeof(metadata_t));
    file_index.read(reinterpret_cast<char *>(&metaData.n_value),
                    sizeof(metadata_t));
    if (metaData.n_row <= 0 || metaData.n_col <= 0 || metaData.n_value <= 0) {
        std::cerr << "Error: Invalid matrix dimensions" << std::endl;
        return metaData;
    }
    if (set_meta != nullptr) {
        set_meta(metaData);
    }

    indptr_t *index_pointers = new indptr_t[metaData.n_row + 1];
    file_index.read(reinterpret_cast<char *>(index_pointers),
                    (metaData.n_row + 1) * sizeof(indptr_t));

    std::streampos pos = file_index.tellg();
    std::streamoff offset = metaData.n_value * sizeof(uint32_t);
    file_data.seekg(pos + offset);

    int start = 0;
    for (int start = 0; start < metaData.n_row; start += batch_size) {
        int end = std::min(start + batch_size, (int)metaData.n_row);
        auto vectors = std::vector<SparseVector>(end - start);
        for (int i = start; i < end; i++) {
            int size = index_pointers[i + 1] - index_pointers[i];
            vectors[i - start] = SparseVector(size);
            auto indices = std::make_unique<uint32_t[]>(size);
            auto values = std::make_unique<float[]>(size);
            file_index.read(reinterpret_cast<char *>(indices.get()),
                            size * sizeof(uint32_t));
            file_data.read(reinterpret_cast<char *>(values.get()),
                           size * sizeof(float));
            for (int j = 0; j < size; j++) {
#ifdef USE_FLOAT
                vectors[i - start][j] = {static_cast<term_t>(indices[j]),
                                         values[j]};
#else
                vectors[i - start][j] = {static_cast<term_t>(indices[j]),
                                         mapPositiveFloatToUint(values[j])};
#endif
            }
        }
        if (callback != nullptr) {
            if (IS_DEBUG) {
                // if debug mode, exit quickly to verify
                callback(std::move(vectors), end - start, start, true);
            } else {
                callback(std::move(vectors), end - start, start,
                         end == metaData.n_row);
            }
        }
        if (IS_DEBUG) {
            // if debug mode, exit quickly to verify
            break;
        }
    }

    file_data.close();
    file_index.close();
    return metaData;
}