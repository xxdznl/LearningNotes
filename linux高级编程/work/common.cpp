#include "common.h"
#define ALIGN_UP_N_BYTES(size, n) ((((size) + (n) - 1) / (n)) * (n))
//64位机器，最小字长8byte
#define ALIGN_UP_8_BYTES(size) ALIGN_UP_N_BYTES(size, 8)
//mmap以4KB为最小单位申请
#define ALIGN_UP_4KB(size) ALIGN_UP_N_BYTES(size, 4096)
//buddy空闲内存块数组链表
BuddyNode * ListFree[15];//ListFree[0]保存16B 2^4大小的块  ListFree[14]保存256KB 2^18大小的块
//buddy已分配内存块链表
Memory * ListAlloc;
pthread_mutex_t* coutMutex;
//Locker* locker;
//求幂
int pow(int m,int n)
{
	int i, p;
    p = 1;
    for(i = 1; i <= n; i++)
        p = p * m;
    return p;
};
//初始化两链表, coutMutex
void initList()
{
    ListAlloc =new Memory(0,0);//头结点
    ListAlloc->pre = nullptr;
    ListAlloc->next = nullptr;
    for(int i = 0;i<=14;i++)
    {
        ListFree[i] = new BuddyNode;
        ListFree[i]->head =new Memory(0,0);
        ListFree[i]->size = pow(2,(i+4));
    }
	//cout互斥
	coutMutex = new pthread_mutex_t;
	if (pthread_mutex_init(coutMutex,0)!= 0) {
        delete coutMutex;
    }
	//读写锁
	//locker = new Locker();
};
//查看空闲块情况
void showListFree()
{
	pthread_mutex_lock(coutMutex);
	std::cout<<"|----------------------------------------------------|"<<std::endl;
	std::cout<<"|FreeList空闲块链表数组                               |"<<std::endl;
	for(int i = 14;i>=0;i--)
	{
		if(ListFree[i]->head->next !=nullptr)
			std::cout<<"|--"<<"第"<<i<<"层，本层块大小"<<ListFree[i]->size<<"B----------------------------|"<<std::endl;
		else
			continue;
		Memory* temp = ListFree[i]->head;
		int count =0;
		while (temp->next!=nullptr)
		{   
			count++;
			temp = temp->next;
			std::cout<<"|----第"<<count<<"块,地址："<<temp->address<<"，块大小："<<temp->size<<"B|"<<std::endl;
		}
		std::cout<<"|----------------------------------------------------|"<<std::endl<<std::endl;
	}
	pthread_mutex_unlock(coutMutex);
};
void showListAlloc()
{
	pthread_mutex_lock(coutMutex);
	std::cout<<"|----------------------------------------------------|"<<std::endl;
	std::cout<<"|FreeAlloc已分配块链表                                |"<<std::endl;
	Memory* temp = ListAlloc;
	int count =0;
	while (temp->next!=nullptr)
	{   
		count++;
		temp = temp->next;
		std::cout<<"|---第"<<count<<"块,地址："<<temp->address<<"，块大小："<<temp->size<<"B|"<<std::endl;
	}
	std::cout<<"|----------------------------------------------------|"<<std::endl<<std::endl;;
	pthread_mutex_unlock(coutMutex);
};
//当前大小应该在第几层去找
int BELONG_TO_N_LAYER(size_t size)
{	
	if(size <= 16)
		return 0;
	int n = 1;
	int two = 16;int twoNext = 32;
	while (1)
	{
		if((size >= two)&&(size <=twoNext))
		{
			return n;
		}
		two = twoNext;
		twoNext = two*2;
		n++;
	}
};
//分配的内存块，插入已分配块链表
int addToListAlloc(Memory * allocMemory,void ** address)
{
	*address = allocMemory->address;
	Memory * findLast = ListAlloc;
	//对所有访问ListFree或者ListAlloc的地方上读锁
	//locker->lockRead();
	while(findLast->next != nullptr)
		findLast = findLast->next;
	//locker->unlockRead();
	//对所有修改ListFree或者ListAlloc的地方上写锁
	//locker->lockWrite();
	//是否可以修改当前结点，给要修改结点上锁
	//if(pthread_mutex_trylock(&(findLast)->mutex) == 0)
	{
		findLast->next = allocMemory;
		//allocMemory可以不用上锁，分配出的块如果做好互斥一定只有一个
		allocMemory->next = nullptr;
		allocMemory->pre = findLast;
		//pthread_mutex_unlock(&(findLast)->mutex);
	}
	//locker->unlockWrite();
	return 0;
};
Memory * MallocBymmap(size_t size)
{
	//以4KB为最小单位申请，向上取整
	size = ALIGN_UP_4KB(size);
	//返回申请到的内存起始地址
	void *addr = mmap(NULL, size, PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if(addr == MAP_FAILED)
	{
		//申请失败，内存不足
		printf("MallocBymmap申请大小{%ld}B失败\n",size);
		return NULL;
	}
	Memory *p = new Memory(addr,size);
	p->FromMmapBit = 1;
	return p;  
}; 