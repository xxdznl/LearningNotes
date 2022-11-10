#include "CLThread.h"
#include "common.h"
CLThread::CLThread(CLExecutiveFunctionProvider *pExecutiveFunctionProvider) : CLExecutive(pExecutiveFunctionProvider)
{
	m_pContext = 0;
	m_bThreadCreated = false;
}
CLThread::~CLThread() {
}

void* CLThread::StartFunctionOfThread(void *pThis){

    CLThread* pThreadThis=(CLThread *)pThis;
	pthread_mutex_lock(coutMutex);
    std::cout<<std::endl<<"线程ID："<<pThreadThis->m_ThreadID<<std::endl;
	pthread_mutex_unlock(coutMutex);
    pThreadThis->m_pExecutiveFunctionProvider->RunExecutiveFunction(pThreadThis->m_pContext);
    return 0;
}

void CLThread::Run(void *pContext){
	if(m_bThreadCreated)
		return ;

	m_pContext = pContext;

	int r = pthread_create(&m_ThreadID, 0, StartFunctionOfThread, this);
	
	if(r != 0)
	{
        std::cout<<std::endl<<"In CLThread"<<this->m_ThreadID<<"::Run(), pthread_create error"<<std::endl;
		delete this;
		return ;
	}

	m_bThreadCreated = true;

	return ;
}

void CLThread::WaitForDeath(){
	if(!m_bThreadCreated)
		return ;

	int r = pthread_join(m_ThreadID, 0);
	if(r != 0)
	{
		std::cout<<std::endl<<"In CLThread"<<this->m_ThreadID<<"::WaitForDeath(), pthread_join error"<<std::endl;
		return ;
	}

	delete this;

	return ;
}




