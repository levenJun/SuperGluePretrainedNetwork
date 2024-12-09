#include <Interpreter/Utils/UtilsBase.h>

namespace rtabmap
{

// return the directory path of the file
std::string UtilsBase::getDir(const std::string & filePath)
{
	std::string dir = filePath;
	int i=(int)dir.size()-1;
	for(; i>=0; --i)
	{
		if(dir[i] == '/' || dir[i] == '\\')
		{
			//remove separators...
			dir[i] = 0;
			--i;
			while(i>=0 && (dir[i] == '/' || dir[i] == '\\'))
			{
				dir[i] = 0;
				--i;
			}
			break;
		}
		else
		{
			dir[i] = 0;
		}
	}

	if(i<0)
	{
		dir = ".";
	}
	else
	{
		dir.resize(i+1);
	}

	return dir;
}

std::string UtilsBase::getName(const std::string & filePath)
{
	std::string fullPath = filePath;
	std::string name;
	for(int i=(int)fullPath.size()-1; i>=0; --i)
	{
		if(fullPath[i] == '/' || fullPath[i] == '\\')
		{
			break;
		}
		else
		{
			name.insert(name.begin(), fullPath[i]);
		}
	}
	return name;
}


std::string UtilsBase::uFormatv (const char *fmt, va_list args)
{
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 1024;
    std::vector<char> dynamicbuf(size);
    char *buf = &dynamicbuf[0];

    va_list argsTmp;

    while (1) {
#if defined(_WIN32) && !defined(__MINGW32__)
	argsTmp = args;
#else
	va_copy(argsTmp, args);
#endif

        // Try to vsnprintf into our buffer.
#ifdef _MSC_VER
    	int needed = vsnprintf_s(buf, size, size, fmt, argsTmp);
#else
    	int needed = vsnprintf (buf, size, fmt, argsTmp);
#endif
    	va_end(argsTmp);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.
        if (needed < (int)size-1 && needed >= 0) {
            // It fit fine so we're done.
            return std::string (buf, (size_t) needed);
        }

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = needed>=0?needed+2:size*2;
        dynamicbuf.resize (size);
        buf = &dynamicbuf[0];
    }
    return std::string(); // would not reach this, but for compiler complaints...
}


std::string UtilsBase::uFormat (const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	std::string buf = uFormatv(fmt, args);
	va_end(args);
    return buf;
}

}