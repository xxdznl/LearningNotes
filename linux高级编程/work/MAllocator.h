#ifndef MALLOCATOR_H
#define MALLOCATOR_H
#include "common.h"
#include "buddy.h"
class MAllocator
{
public:
    static MAllocator *getInstance();//获取MAllocator对象指针
    void* MyMalloc(size_t size,int threadCount);//申请空间
    void MyFree(void* address,int threadCount); //释放空间
private:
    MAllocator();
    ~MAllocator();
    static pthread_mutex_t *getCreatingMAllocatorMutex();//获取创建内存分配器的互斥量
private:
    
    //用于MAllocator实例的创建，保证只创建一个MAllocator实例
    static pthread_mutex_t *m_pMutexForCreatingMAllocator;
    static MAllocator* m_MAllocator;


};
#endif