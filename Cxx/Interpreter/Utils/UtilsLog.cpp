#include <Interpreter/Utils/UtilsLog.h>
#include <sys/time.h>
#include <cassert>  // 确保包含此头文件

#define COLOR_NORMAL "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"

void ULogger::write(ULogger::Level level,
		const char * file,
		int line,
		const char * function,
		const char* msg,
		...)
{

    std::string time = "";
    if(true)
    {
        getTime(time);
        time.append(" - ");
    }

    if(true){
        _writeStr(time.c_str());
    }    

    switch (level)
    {
    case kDebug:
        _writeStr("_D_");
        break;
    case kInfo:
        _writeStr("_I_");
        break;
    case kWarning:
        _writeStr("_WW_");
        break;
    case kError:
    case kFatal:
        _writeStr("_EE_");
        break;
    default:
        break;
    }

    va_list args;
    va_start(args, msg);
    if(true){
        _write(msg, args);        
    }
    va_end(args);

    if(true)
    {
        std::string endline = "\r\n";        
        _writeStr(endline.c_str());
    }

    if(level >= kFatal){
        assert(false);
    }

}


int ULogger::getTime(std::string &timeStr)
{
    struct tm timeinfo;
    const int bufSize = 30;
    char buf[bufSize] = {0};

#if _MSC_VER
    time_t rawtime;
    time(&rawtime);
    localtime_s (&timeinfo, &rawtime );
    int result = sprintf_s(buf, bufSize, "%d-%s%d-%s%d %s%d:%s%d:%s%d",
        timeinfo.tm_year+1900,
        (timeinfo.tm_mon+1) < 10 ? "0":"", timeinfo.tm_mon+1,
        (timeinfo.tm_mday) < 10 ? "0":"", timeinfo.tm_mday,
        (timeinfo.tm_hour) < 10 ? "0":"", timeinfo.tm_hour,
        (timeinfo.tm_min) < 10 ? "0":"", timeinfo.tm_min,
        (timeinfo.tm_sec) < 10 ? "0":"", timeinfo.tm_sec);
#elif WIN32
    time_t rawtime;
    time(&rawtime);
    timeinfo = *localtime (&rawtime);
    int result = snprintf(buf, bufSize, "%d-%s%d-%s%d %s%d:%s%d:%s%d",
		timeinfo.tm_year+1900,
		(timeinfo.tm_mon+1) < 10 ? "0":"", timeinfo.tm_mon+1,
		(timeinfo.tm_mday) < 10 ? "0":"", timeinfo.tm_mday,
		(timeinfo.tm_hour) < 10 ? "0":"", timeinfo.tm_hour,
		(timeinfo.tm_min) < 10 ? "0":"", timeinfo.tm_min,
		(timeinfo.tm_sec) < 10 ? "0":"", timeinfo.tm_sec);
 #else
    struct timeval rawtime;
    gettimeofday(&rawtime, NULL);
    localtime_r (&rawtime.tv_sec, &timeinfo);
	int result = snprintf(buf, bufSize, "%d-%s%d-%s%d %s%d:%s%d:%s%d.%s%d",
		timeinfo.tm_year+1900,
		(timeinfo.tm_mon+1) < 10 ? "0":"", timeinfo.tm_mon+1,
		(timeinfo.tm_mday) < 10 ? "0":"", timeinfo.tm_mday,
		(timeinfo.tm_hour) < 10 ? "0":"", timeinfo.tm_hour,
		(timeinfo.tm_min) < 10 ? "0":"", timeinfo.tm_min,
		(timeinfo.tm_sec) < 10 ? "0":"", timeinfo.tm_sec,
	    (rawtime.tv_usec/1000) < 10 ? "00":(rawtime.tv_usec/1000) < 100?"0":"", int(rawtime.tv_usec/1000));
#endif
    if(result)
    {
        timeStr.append(buf);
    }
    return result;
}

void ULogger::_write(const char* msg, va_list arg)
{
    vprintf(msg, arg);
}

void ULogger::_writeStr(const char* msg)
{
    printf("%s", msg);
}