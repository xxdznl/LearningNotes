#ifndef COMMON_H
#define COMMON_H
#include "stdlib.h"
#include "sys/mman.h"
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#define THRESHOLD_FOR_MMAP (256 * 1024)
#define SEGMENT_SIZE (2 * 1024 * 1024) //一次mmap2MB大小作为堆空间
//求指数
int pow(int m,int n);
//当前大小应该在第几层去找
int BELONG_TO_N_LAYER(size_t size);
//参数信息
struct Args {
    int max;
	int min;
	int size;
	int threadCount;
    Args(int _max, int _min, int _size, int _threadCount) : max(_max), min(_min), size(_size), threadCount(_threadCount) {}
};
//内存块地址信息结构体
typedef struct Memory
{
	void * address;//内存块地址
	unsigned long size;//内存块大小
	unsigned long FromMmapBit;//是否是从mmap直接申请的
	Memory* pre;//前一内存块
	Memory* next;//后一内存块
	pthread_mutex_t mutex;//本块互斥访问
    Memory(void * _address, size_t _size) : address(_address), size(_size) 
    {
		FromMmapBit = 0;
		pre = nullptr;
		next = nullptr;
        pthread_mutex_init(&mutex, NULL);
    }
}Memory;
typedef struct BuddyNode
{
	unsigned long size;
	Memory* head;
}BuddyNode;
//buddy空闲内存块数组链表
extern BuddyNode * ListFree[15];//ListFree[0]保存16B 2^4大小的块  ListFree[14]保存256KB 2^18大小的块
//buddy已分配内存块链表
extern Memory * ListAlloc;
//初始化ListFree和ListAlloc
void initList();
//进程结束，销毁所有结点
void clearList();

void showListFree();
void showListAlloc();
//分配的内存块，插入已分配块链表
int addToListAlloc(Memory * allocMemory,void** address);
Memory * MallocBymmap(size_t size); 
//输出时临界区，不然cout乱序输出
extern pthread_mutex_t* coutMutex;
//读写锁
//extern Locker* locker;
#endif