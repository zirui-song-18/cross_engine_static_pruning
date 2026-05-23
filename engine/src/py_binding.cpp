#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <ranges>
#include <string>
#include <thread>
#include <vector>

#include "AlphaMassDoc_AlphaMassPosting.h"
#include "AlphaMassQuery.h"
#include "Cluster.h"
#include "CsrReader.h"
#include "DataTypes.h"
#include "LinscanIndex.h"
#include "Serializer.h"
#include "TypeConvert.h"
#include "distance.h"

namespace py = pybind11;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(sparse_engine, m) {
    m.doc() = "Sparse retrieval engine with pruning support";

    py::class_<LinscanIndex>(m, "LinscanIndex")
        .def(py::init<>())
        .def("load", &LinscanIndex::Load)
        .def("search", &LinscanIndex::Search)
        .def("get_memory_usage", &LinscanIndex::GetMemoryUsage);

    py::class_<AlphaMassQuery>(m, "AlphaMassQuery")
        .def(py::init<>())
        .def("load", &AlphaMassQuery::Load)
        .def("search", &AlphaMassQuery::Search)
        .def("get_memory_usage", &AlphaMassQuery::GetMemoryUsage)
        .def("set_window_size", &AlphaMassQuery::SetWindowSize)
        .def("get_window_size", &AlphaMassQuery::GetWindowSize);

    // py::class_<AlphaMassQueryWindowed>(m, "AlphaMassQueryWindowed")
    //     .def(py::init<>())
    //     .def(py::init<size_t>(), py::arg("window_size"))
    //     .def("load", &AlphaMassQueryWindowed::Load)
    //     .def("search", &AlphaMassQueryWindowed::Search)
    //     .def("get_memory_usage", &AlphaMassQueryWindowed::GetMemoryUsage)
    //     .def("set_window_size", &AlphaMassQueryWindowed::SetWindowSize)
    //     .def("get_window_size", &AlphaMassQueryWindowed::GetWindowSize);

    py::class_<AlphaMassDoc_AlphaMassPosting>(m,
                                              "AlphaMassDoc_AlphaMassPosting")
        .def(py::init<>())
        .def("load", &AlphaMassDoc_AlphaMassPosting::Load)
        .def("load_alpha_mass", &AlphaMassDoc_AlphaMassPosting::LoadAlphaMass,
             py::arg("filename"), py::arg("batch_size"),
             py::arg("alpha_prune_ratio"), py::arg("list_alpha_prune_ratio"),
             py::arg("idf_prune_percent") = 0.0f,
             py::arg("doc_max_ratio") = -1.0f, py::arg("doc_fixed_top") = -1,
             py::arg("list_max_ratio") = -1.0f, py::arg("list_fixed_top") = -1)
        .def("search", &AlphaMassDoc_AlphaMassPosting::Search)
        .def("get_memory_usage",
             &AlphaMassDoc_AlphaMassPosting::GetMemoryUsage)
        .def("write_per_query_latencies",
             &AlphaMassDoc_AlphaMassPosting::WritePerQueryLatencies);

    py::class_<QueryArguments>(m, "QueryArguments")
        .def(py::init([](int k, int kprime, int cut, float hf) {
            return new QueryArguments(k, kprime, cut, hf);
        }))
        .def(py::init([](int k, int kprime, int cut, float hf, int ip_budget) {
            return new QueryArguments(k, kprime, cut, hf, ip_budget);
        }))
        .def(py::init([](int k, int kprime, int cut, float hf, int ip_budget,
                         int doc_limit) {
            return new QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit);
        }))
        .def(py::init([](int k, int kprime, int cut, float hf, int ip_budget,
                         int doc_limit, int num_worker) {
            return new QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit,
                                      num_worker);
        }))
        .def(py::init([](int k, int kprime, int cut, float hf, int ip_budget,
                         int doc_limit, int num_worker,
                         float alpha_prune_ratio) {
            return new QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit,
                                      num_worker, alpha_prune_ratio);
        }))
        .def(py::init([](int k, int kprime, int cut, float hf, int ip_budget,
                         int doc_limit, int num_worker, float alpha_prune_ratio,
                         float max_ratio, int fixed_top) {
            return new QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit,
                                      num_worker, alpha_prune_ratio, max_ratio,
                                      fixed_top);
        }))
        .def_readonly("k", &QueryArguments::k)
        .def_readonly("kprime", &QueryArguments::kprime)
        .def_readonly("cut", &QueryArguments::cut)
        .def_readonly("hf", &QueryArguments::heap_factor)
        .def_readonly("num_worker", &QueryArguments::num_worker)
        .def_readonly("alpha_prune_ratio", &QueryArguments::alpha_prune_ratio)
        .def_readonly("max_ratio", &QueryArguments::max_ratio)
        .def_readonly("fixed_top", &QueryArguments::fixed_top);

    m.def("add", &add, "A function that adds two numbers");

#ifdef USE_FLOAT
    m.def("dp",
          static_cast<float (*)(const SparseVector &, const SparseVector &)>(
              &dot_product),
          "A dp function between sparse and sparse");
#else
    m.def("dp",
          static_cast<RET_T (*)(const SparseVector &, const SparseVector &)>(
              &dot_product),
          "A dp function between sparse and sparse");
    m.def("dp_SD",
          static_cast<RET_T (*)(const SparseVector &, const DenseVector &)>(
              &dot_product),
          "A dp function between sparse and dense");
#endif
}