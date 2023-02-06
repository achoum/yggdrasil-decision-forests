#ifndef DLL_H
#define DLL_H

#ifdef COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT void hello();
extern "C" DLLEXPORT void run_ydf();

#endif