#ifndef __PARALLELIZATION_H__
#define __PARALLELIZATION_H__
#include <future>
#include <iostream>
#include <vector>

class Parallelization {
    static unsigned int GetWorkers(int threads) {
        const unsigned int N_OPTIMAL_THREADS =
            std::max(1u, std::thread::hardware_concurrency());
        return std::min(N_OPTIMAL_THREADS, static_cast<unsigned int>(threads));
    }

    static int GetBatchSize(int size, int workers) {
        return (size + workers - 1) / workers;
    }

  public:
    template <typename Ret, typename Func, typename ClassPtr, typename... Args>
    inline static std::vector<std::future<Ret>>
    Run(size_t max_threads, size_t total, Func &&func, ClassPtr instance,
        Args &&...args) {
        const unsigned int N_WORKERS = GetWorkers(max_threads);
        int batch_size = GetBatchSize(total, N_WORKERS);
        std::vector<std::future<Ret>> futures;
        futures.reserve(N_WORKERS);
        for (size_t i = 0; i <= N_WORKERS; ++i) {
            int start = i * batch_size;
            int end = std::min((i + 1) * batch_size, total);
            if (start >= end)
                continue;
            futures.push_back(
                std::async(std::launch::async, std::forward<Func>(func),
                           instance, start, end, std::forward<Args>(args)...));
        }
        return futures;
    }

    template <typename Func, typename ClassPtr, typename... Args>
    inline static std::vector<std::future<void>>
    Run(size_t max_threads, size_t total, Func &&func, ClassPtr instance,
        Args &&...args) {
        const unsigned int N_WORKERS = GetWorkers(max_threads);
        int batch_size = GetBatchSize(total, N_WORKERS);
        std::vector<std::future<void>> futures;
        futures.reserve(N_WORKERS);
        for (size_t i = 0; i <= N_WORKERS; ++i) {
            int start = i * batch_size;
            int end = std::min((i + 1) * batch_size, total);
            if (start >= end)
                continue;
            futures.push_back(
                std::async(std::launch::async, std::forward<Func>(func),
                           instance, start, end, std::forward<Args>(args)...));
        }
        return futures;
    }
};

#endif