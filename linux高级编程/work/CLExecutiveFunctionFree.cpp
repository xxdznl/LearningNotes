#include "CLExecutiveFunctionFree.h"
CLExecutiveFunctionFree::CLExecutiveFunctionFree()
{

};
CLExecutiveFunctionFree::~CLExecutiveFunctionFree()
{

};
void CLExecutiveFunctionFree::RunExecutiveFunction(void* pContext)
{
    //std::cout<<"Freeeeeeeeeeee"<<std::endl;
    struct Args* args = (struct Args*)pContext;//参数
    void* address = nullptr;//块地址
    //sleep(1);//
    for(int i = 0;;i++)
    {
        if(ListAlloc->next == nullptr)
        {
            break;
        }    
        Memory * isEmpty = ListAlloc->next;//取第一个分配块进行释放 //记得互斥访问
        if(isEmpty!=nullptr)
        {
            //std::cout<<"Freeeeeeeeeeee"<<std::endl;
            address = isEmpty->address;
            MAllocator * Allocator = MAllocator::getInstance();
            Allocator->MyFree(address,args->threadCount);
            //每FREE 20次查看一次结果
            if((i+1)%10==0)
            {
                 //showListFree();
                 //showListAlloc();
            }
        }
    }
    //showListFree();
    //showListAlloc();
}