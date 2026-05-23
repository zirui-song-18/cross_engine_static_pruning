#ifndef __TYPE_CONVERT_H__
#define __TYPE_CONVERT_H__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <type_traits>

#include "types.h"

#ifdef USE_FLOAT
#define MAP_VALUE_TYPE(x) x
#else
#define MAP_VALUE_TYPE(x) mapPositiveFloatToUint(x)

inline u_value_t mapPositiveFloatToUint(float value, float maxValue = 3.0f) {
    // Ensure the value is within the specified range
    value = std::max(0.0f, std::min(maxValue, value));

    // Scale the value to fit in the u_value_t range
    float scaled = (value / maxValue) * std::numeric_limits<u_value_t>::max();

    // Round to nearest integer and cast to u_value_t
    return static_cast<u_value_t>(std::round(scaled));
}

inline int8_t mapFloatToInt8Signed(float value, float maxAbsValue) {
    maxAbsValue = std::abs(maxAbsValue);
    value = std::max(-maxAbsValue, std::min(maxAbsValue, value));
    // map to range [-128, 127]
    return static_cast<int8_t>(std::round(value * 127.0f / maxAbsValue));
}

inline int8_t mapFloatToInt8Positive(float value, float maxValue) {
    value = std::max(0.0f, std::min(maxValue, value));
    // map to range [0, 127]
    return static_cast<int8_t>(std::round(value * 127.0f / maxValue));
}
// Base template for different types - will assert
template <typename R, typename T> inline R OneToTheOther(T value) {
    static_assert(std::is_same<R, T>::value,
                  "Types must be the same unless explicitly specialized");
    return value;
}

// Specialization for float to u_value_t conversion
template <> inline u_value_t OneToTheOther<u_value_t, float>(float value) {
    return mapPositiveFloatToUint(value);
}

// Specialization for double to u_value_t conversion
template <> inline u_value_t OneToTheOther<u_value_t, double>(double value) {
    return mapPositiveFloatToUint(value);
}

#endif

#endif