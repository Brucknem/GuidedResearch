//
// Created by brucknem on 18.01.21.
//

#include "TimeMeasurable.hpp"

providentia::utils::TimeMeasurable::TimeMeasurable(std::string name, int verbosity) : name(std::move(name)),
                                                                                      verbosity(verbosity) {}

void providentia::utils::TimeMeasurable::setName(std::string _name) {
    name = std::move(_name);
}

void providentia::utils::TimeMeasurable::setVerbosity(int _verbosity) {
    verbosity = _verbosity;
}

void providentia::utils::TimeMeasurable::setNameAndVerbosity(std::string _name, int _verbosity) {
    setName(std::move(_name));
    setVerbosity(_verbosity);
}

std::chrono::milliseconds providentia::utils::TimeMeasurable::now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
}

void providentia::utils::TimeMeasurable::clear() {
    timestamps.clear();
    durations.clear();
    previous = now();
    addTimestamp("Start");
}

void providentia::utils::TimeMeasurable::addTimestamp(const std::string &measurementName, int minVerbosity) {
    if (verbosity < minVerbosity) {
        return;
    }
    auto now = providentia::utils::TimeMeasurable::now();
    timestamps.emplace_back(measurementName, now);
    durations.emplace_back(std::make_pair(measurementName, (now - previous).count()));
    previous = providentia::utils::TimeMeasurable::now();
}

long providentia::utils::TimeMeasurable::getTotalMilliseconds() {
    return (timestamps[timestamps.size() - 1].second - timestamps[0].second).count();
}

std::string providentia::utils::TimeMeasurable::durations_str() {
    std::stringstream ss;

    for (const auto &duration : durations) {
        ss << "[" << name << "] " << duration.second << "ms : " << duration.first << std::endl;
    }

    return ss.str();
}

providentia::utils::TimeMeasurable::~TimeMeasurable() = default;





