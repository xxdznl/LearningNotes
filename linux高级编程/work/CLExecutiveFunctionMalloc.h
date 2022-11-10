#ifndef CLEXECUTIVEFUNCTIONMALLOC_H
#define CLEXECUTIVEFUNCTIONMALLOC_H

#include "CLExecutiveFunctionProvider.h"
#include "MAllocator.h"

class CLExecutiveFunctionMalloc: public CLExecutiveFunctionProvider{
public:
    CLExecutiveFunctionMalloc();
	virtual ~CLExecutiveFunctionMalloc();
public:
    void RunExecutiveFunction(void* pContext);
    
private:
};
#endif