#pragma once
#include <iostream>
#include <sstream>
#include <vector>


/// Prints  a vector in the format [item0, item1, item2]
template <typename T>
void printVec(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << '[';
    auto size = vec.size();
    for (int i = 0; i < size; ++i) {
        ss << vec[i];
        if (i < size - 1) {
            ss << ", ";
        }
    }
    ss << ']';
    std::cout << ss.str();
}


template <typename T>
std::ostream& operator<< (std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (auto && item : v) {
        os << " " << item;
    }os << " ]";
    return os;
}