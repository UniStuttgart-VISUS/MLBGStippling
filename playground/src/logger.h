#ifndef LOGGER_H
#define LOGGER_H

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

static FILE* __logger_file = nullptr;

inline void logf(const char* format, ...) {
    if (!__logger_file) {
        __logger_file = fopen("playground.log.json", "w+");
        assert(__logger_file && "Cannot open log file");
    }
    va_list arglist;
    va_start(arglist, format);
    vfprintf(__logger_file, format, arglist);
    va_end(arglist);
}

inline void logflush() {
    if (!__logger_file) {
        return;
    }
    fflush(__logger_file);
}

#endif
