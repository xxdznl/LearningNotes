#ifndef CLEXECUTIVEFUNCTIONFREE_H
#define CLEXECUTIVEFUNCTIONFREE_H
#include "MAllocator.h"
#include "CLExecutiveFunctionProvider.h"
class CLExecutiveFunctionFree: public CLExecutiveFunctionProvider{
public:
    CLExecutiveFunctionFree();
    virtual ~CLExecutiveFunctionFree();
public:

    void RunExecutiveFunction(void* pContext);
private:
   
};
#endif