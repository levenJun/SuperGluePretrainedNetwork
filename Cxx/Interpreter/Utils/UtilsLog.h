#pragma once

#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <stdarg.h>

#define ULOGGER_LOG(level, ...) ULogger::write(level, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

#define ULOGGER_DEBUG(...)   ULOGGER_LOG(ULogger::kDebug,   __VA_ARGS__)
#define ULOGGER_INFO(...)    ULOGGER_LOG(ULogger::kInfo,    __VA_ARGS__)
#define ULOGGER_WARN(...) 	 ULOGGER_LOG(ULogger::kWarning, __VA_ARGS__)
#define ULOGGER_ERROR(...)   ULOGGER_LOG(ULogger::kError,   __VA_ARGS__)
#define ULOGGER_FATAL(...)   ULOGGER_LOG(ULogger::kFatal,   __VA_ARGS__) // Throw UException

#define UDEBUG(...)   ULOGGER_DEBUG(__VA_ARGS__)
#define UINFO(...)    ULOGGER_INFO(__VA_ARGS__)
#define UWARN(...) 	  ULOGGER_WARN(__VA_ARGS__)
#define UERROR(...)   ULOGGER_ERROR(__VA_ARGS__)
#define UFATAL(...)   ULOGGER_FATAL(__VA_ARGS__) // Throw UException

// Throw UException
#define UASSERT(condition) if(!(condition)) ULogger::write(ULogger::kFatal, __FILE__, __LINE__, __FUNCTION__, "Condition (%s) not met!", #condition)
#define UASSERT_MSG(condition, msg_str) if(!(condition)) ULogger::write(ULogger::kFatal, __FILE__, __LINE__, __FUNCTION__, "Condition (%s) not met! [%s]", #condition, msg_str)


class ULogger{

public:

    enum Type{kTypeNoLog, kTypeConsole, kTypeFile};
    enum Level{kDebug, kInfo, kWarning, kError, kFatal};

    /*
     * Write a message to logger: use UDEBUG(), UINFO(), UWARN(), UERROR() or UFATAL() instead.
     * @param level the log level of this message
     * @param file the file path
     * @param line the line in the file
     * @param function the function name in which the message is logged
     * @param msg the message to write
     * @param ... the variable arguments
     */
    static void write(ULogger::Level level,
    		const char * file,
    		int line,
    		const char *function,
    		const char* msg,
    		...);

    /**
     * Get the time in the format "2008-7-13 12:23:44".
     * @param timeStr string were the time will be copied.
     * @return the number of characters written, or 0 if an error occurred.
     */
    static int getTime(std::string &timeStr);

    /*
     * Write a message on the output with the format :
     * "A message". Inherited class
     * must override this method to output the message. It 
     * does nothing by default.
     * @param msg the message to write.
     * @param arg the variable arguments
     */
    static void _write(const char*, va_list); // Do nothing by default
    static void _writeStr(const char* msg);

};