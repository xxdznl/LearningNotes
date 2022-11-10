#include "CLExecutiveFunctionMalloc.h"
CLExecutiveFunctionMalloc::CLExecutiveFunctionMalloc()
{

};
CLExecutiveFunctionMalloc::~CLExecutiveFunctionMalloc()
{

};
void CLExecutiveFunctionMalloc:: RunExecutiveFunction(void* pContext)
{
    struct Args* args = (struct Args*)pContext;//线程参数
    for(int i = 0; i <args->size ; i++)
	{
        int mem_size = rand() % args->max + args->min;
        MAllocator * Allocator = MAllocator::getInstance();
        Allocator->MyMalloc(mem_size,args->threadCount);
    }
   
};