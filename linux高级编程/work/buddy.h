#ifndef BUDDY_H
#define BUDDY_H
#include "common.h"
//是否存在空闲buddy内存块 =1意味着存在空闲块，根据申请大小判断，直接从对应大小层往上找
int IsListFreeEmpty(int size);
//空闲buddy内存块不够（初次申请内存，或者多次分配后无可用内存），找不到空闲buddy内存块。
//mmap一个2M大小的空间，并拆分为8个256kb，buddy块，插入ListFree[14] 额 不如直接mmap8个256kb的块
void BuddyBlockAllocMmap2MB();
Memory* BuddyAllocBlock(size_t size);
//合并到空闲块链表
int BuddyMergeToFreeList(Memory *MemoryToMerge);
#endif