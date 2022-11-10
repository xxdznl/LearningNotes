#include "MAllocator.h"
#include "common.h"
#include "buddy.h"
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLExecutiveFunctionMalloc.h"
#include "CLExecutiveFunctionFree.h"
int main()
{ 
    initList();

    //方案一  单线程申请5块，大小16B,256B，16KB，128KB，257KB
    // MAllocator* test1 = MAllocator::getInstance();
    // test1->MyMalloc(16,1);
    // test1->MyMalloc(256,1);
    // test1->MyMalloc(16 * 1024,1);
    // test1->MyMalloc(128 * 1024,1);
    // test1->MyMalloc(257 * 1024,1);
    // showListFree();
    // showListAlloc();


    //方案二  单线程申请3块，大小256B，128KB，257KB, 同时开始回收内存。
    // MAllocator* test1 = MAllocator::getInstance();
    // void * addr1 = test1->MyMalloc(256,1);
    // void * addr2 = test1->MyMalloc(128 * 1024,1);
    // void * addr3 = test1->MyMalloc(257 * 1024,1);
    // test1->MyFree(addr1,1);test1->MyFree(addr2,1);test1->MyFree(addr3,1);
    // showListFree();
    // showListAlloc();

    //方案三  大规模 多线程（伪）申请释放内存
    // CLExecutive *pThreadMalloc=new CLThread(new CLExecutiveFunctionMalloc());
    // CLExecutive *pThreadMalloc2=new CLThread(new CLExecutiveFunctionMalloc());
    // CLExecutive *pThreadFree=new CLThread(new CLExecutiveFunctionFree());
    // CLExecutive *pThreadFree2=new CLThread(new CLExecutiveFunctionFree());
    // int max = 257 * 1024;//最大申请块范围
    // int min = 16 ;//最小申请块范围
    // int size = 4;//每个线程申请个数
    // int threadCount = 1;
    // pThreadMalloc->Run(new Args(max,min,size,threadCount));
    // pThreadMalloc2->Run(new Args(max,min,size,threadCount+1));
    // pThreadFree->Run(new Args(max,min,size,threadCount));
    // pThreadFree2->Run(new Args(max,min,size,threadCount+1));
    // pThreadMalloc->WaitForDeath();
    // pThreadMalloc2->WaitForDeath();
    // pThreadFree->WaitForDeath();
    // pThreadFree2->WaitForDeath();
    
    getchar();
    return 0;
}