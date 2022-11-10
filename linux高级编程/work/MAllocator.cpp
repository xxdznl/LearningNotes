#include "MAllocator.h"

MAllocator::MAllocator()
{
	
}
//私有静态变量，只能在内部初始化
pthread_mutex_t* MAllocator::m_pMutexForCreatingMAllocator = MAllocator::getCreatingMAllocatorMutex();
MAllocator * MAllocator::m_MAllocator = 0;
//获取内存分配器的互斥量
pthread_mutex_t *MAllocator::getCreatingMAllocatorMutex() {
    //初始化内存分配器mutex
    pthread_mutex_t *p = new pthread_mutex_t;
    if (pthread_mutex_init(p, 0) != 0) {
        delete p;
        return 0;
    }
    return p;
}
MAllocator::~MAllocator()
{
	//销毁创建Allocator锁
	pthread_mutex_destroy(this->m_pMutexForCreatingMAllocator);
    
}
//获取MAllocator实例
MAllocator *MAllocator::getInstance() {
    /**
    只允许构建一个MAllocator实例
    **/
   	//已经创建实例m_MAllocator
    if(m_MAllocator != 0) {
        return m_MAllocator;
    }
	//创建实例时加锁避免创建多个allocator
    if (pthread_mutex_lock(m_pMutexForCreatingMAllocator) != 0) {
        return 0;
    }
	//实例第一次创建
    if (m_MAllocator == 0) {
        try {
            m_MAllocator = new MAllocator();
        } catch (const char *) {
            pthread_mutex_unlock(m_pMutexForCreatingMAllocator);
            return 0;
        }
    }
	//创建完解锁
    if (pthread_mutex_unlock(m_pMutexForCreatingMAllocator) != 0) {
        return 0;
    }
    return m_MAllocator;
}
//申请空间
void* MAllocator::MyMalloc(size_t size,int threadCount)
{	
	//单线程测试需要返回内存地址
	void* getAddress = nullptr;
	//非法输入
	if(size <= 0)
		return nullptr;
	//判断大小，超过256kb阈值，使用mmap直接分配
	else if(size > THRESHOLD_FOR_MMAP)
	{
		//mmap是多线程安全的，可以在两个线程里同时调用mmap这个函数。不用加锁
		int r =addToListAlloc(MallocBymmap(size),&getAddress);
		if(r == 0)
		{	
			//std::cout<<"mmap直接分配块大小"<<size<<"成功"<<std::endl;
			pthread_mutex_lock(coutMutex);
			std::cout<<"Malloc线程"<<threadCount<<" mmap直接分配块大小"<<size<<"B成功"<<std::endl<<std::endl;
			pthread_mutex_unlock(coutMutex);
			return getAddress;
		}
	} 
	//否则使用buddy空闲块分配
	else
	{
		//是否存在可用buddy内存块 =i意味着第i层存在空闲块，根据申请大小判断，直接从对应大小层往上找
		if(IsListFreeEmpty(size) != 0)
		{
			//使用buddy算法分配空闲块
			int r =addToListAlloc(BuddyAllocBlock(size),&getAddress);
			if(r == 0)
			{	
				//std::cout<<"使用buddy算法分配块大小"<<size<<"B成功"<<std::endl;
				pthread_mutex_lock(coutMutex);
				std::cout<<"Malloc线程"<<threadCount<<" 使用buddy算法分配块大小"<<size<<"B成功"<<std::endl<<std::endl;
				pthread_mutex_unlock(coutMutex);
				return getAddress;
			}
		} 
		//空闲buddy内存块不够（初次申请内存，或者多次分配后无可用内存），找不到空闲buddy内存块。
		else
		{
			//mmap一个2M大小的空间，并拆分为8个256kb，buddy块，插入ListFree[14]
			BuddyBlockAllocMmap2MB();
			//然后再使用buddy算法分配空闲块
			int r =addToListAlloc(BuddyAllocBlock(size),&getAddress);
			if(r == 0)
			{	
				//std::cout<<"块不够了，重新分配2M空间，使用buddy算法分配块大小"<<size<<"B成功"<<std::endl;
				pthread_mutex_lock(coutMutex);
				std::cout<<"Malloc线程"<<threadCount<<" 块不够了，重新分配2M空间，使用buddy算法分配块大小"<<size<<"B成功"<<std::endl<<std::endl;
				pthread_mutex_unlock(coutMutex);
				return getAddress;
			}
		}
	}
	return nullptr;
}
//释放内存块，有兄弟就向上合并。如果是mmap直接分配的，直接释放;threadCount记录第几个线程访问
void MAllocator:: MyFree(void* address,int threadCount)
{
	//找到要释放的块
	Memory* findToFree = ListAlloc;
	Memory* prefindToFree = findToFree->pre;
	//locker->lockRead();
	//std::cout<<findToFree->address<<"--------------2222--------------"<<findToFree->size<<std::endl;
	while(findToFree->next!=nullptr)
	{
		prefindToFree = findToFree;
		//跳过头结点
		findToFree = findToFree->next;
		// std::cout<<findToFree->address<<"--------------1111--------------"<<findToFree->size<<std::endl;
		// std::cout<<"--------------1111--------------"<<address<<std::endl;
		if(findToFree->address == address)
		{
			//locker->unlockRead();
			//locker->lockWrite();
			//std::cout<<"找到地址对应块了"<<std::endl;
			//删除块修改了三个块，当前块，前一块，后一块，都加锁
			//从分配链表中删除
			//if(pthread_mutex_trylock(&(prefindToFree)->mutex)==0 && pthread_mutex_trylock(&(findToFree)->mutex)==0)
			{
				prefindToFree->next = findToFree->next;
				//还剩下不止一块时，下一块指向头结点
				if(findToFree->next != nullptr)// && (pthread_mutex_trylock(&(findToFree->next)->mutex)==0))//在
				{
					findToFree->next->pre = prefindToFree;
					//pthread_mutex_unlock(&(findToFree->next)->mutex);
				}
				findToFree->pre = nullptr;
				findToFree->next = nullptr;
				//pthread_mutex_unlock(&(prefindToFree)->mutex);
				//pthread_mutex_unlock(&(findToFree)->mutex);
			}
			//locker->unlockWrite();
			//不是从mmap直接获取的
			if(findToFree->FromMmapBit == 0)
			{
				//std::cout<<"找到地址对应块了"<<std::endl;
				//合并到空闲块链表数组
				int tempSize = findToFree->size;
				pthread_mutex_lock(coutMutex);
				std::cout<<std::endl<<"|-------------------------合并过程-------------------------|"<<std::endl;
				std::cout<<"|Free线程"<<threadCount<<" 开始合并地址为："<<address<<" 大小为："<<tempSize<<"的块"<<std::endl;
				pthread_mutex_unlock(coutMutex);

				int layer = BuddyMergeToFreeList(findToFree);

				pthread_mutex_lock(coutMutex);
				std::cout<<"|Free线程"<<threadCount<<" 成功合并该块到第"<<layer<<"层"<<std::endl;
				std::cout<<"|-------------------------------------------------------------|"<<std::endl<<std::endl;
				pthread_mutex_unlock(coutMutex);
				return;
			}
			//mmap直接获取的直接munmap掉,且对应的Memory块也删除
			else
			{
				int temp = findToFree->size;
				munmap((void*)findToFree->address, findToFree->size);
				pthread_mutex_lock(coutMutex);
				std::cout<<"Free线程"<<threadCount<<" 通过munmap删除--地址为："<<address<<" 大小为："<<temp<<"的块"<<std::endl<<std::endl;
				pthread_mutex_unlock(coutMutex);
				delete findToFree;
				return;
			}
		}
	}
	//locker->unlockRead();
}