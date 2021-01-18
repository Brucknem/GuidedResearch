//
// Created by brucknem on 18.01.21.
//

#include "CSVWriter.hpp"

providentia::utils::CSVWriter::CSVWriter(const std::string &path, bool write) : write(write) {
    if (!write) {
        return;
    }
    fileStream.open(path);
    fileStream.close();
    fileStream.open(path, std::ios_base::app); // append instead of overwrite
}

providentia::utils::CSVWriter::~CSVWriter() {
    if (!write) {
        return;
    }
    fileStream.close();
}

void providentia::utils::CSVWriter::newLine() {
    if (!write) {
        return;
    }
    std::string line = currentLine.str();
    currentLine.str("");
    line.pop_back();
    fileStream << line << std::endl;
}

template<typename Arg, typename... Args>
void providentia::utils::CSVWriter::append(Arg &&arg, Args &&... args) {
    if (!write) {
        return;
    }
    currentLine << std::forward<Arg>(arg);
    using expander = int[];
    (void) expander{0, (void(currentLine << ',' << std::forward<Args>(args)), 0)...};
    currentLine << ",";
}
