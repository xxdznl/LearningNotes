#ifndef CLTHREAD_H
#define CLTHREAD_H
#include<pthread.h>
#include<iostream>
#include "CLExecutive.h"
class CLThread : public CLExecutive
{
public:
    explicit CLThread(CLExecutiveFunctionProvider* pExecutiveFunctionProvider);
    ~CLThread();
    void Run(void *pContext);
    void WaitForDeath();

private:
    static void* StartFunctionOfThread(void* pThis);

private:
    void *m_pContext;
    pthread_t m_ThreadID;

    bool m_bThreadCreated;//线程创建即唯一不能重复创建
};


#endif // CLTHREAD_H
