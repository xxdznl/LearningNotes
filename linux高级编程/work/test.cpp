#include "sys/mman.h"
#include <iostream>
using namespace std;
#define ALIGN_UP_N_BYTES(size, n) (size * (n))
#define ALIGN_UP_8_BYTES(size) ALIGN_UP_N_BYTES(size, 8)
#define ALIGN_UP_4KB(size) ALIGN_UP_N_BYTES(size, 4096)
struct SLMemBlock
{
	unsigned long CurBlkInUseBit : 1;
	unsigned long PrevBlkInUseBit : 1;
	unsigned long FromMmapBit : 1;
	unsigned long ulBlockSize : 61;
};
void *MallocBymmap(size_t size)
{
	//4096 * 256 * 1024  
	size = ALIGN_UP_4KB(size);
	//返回申请到的内存起始地址
	void *addr = mmap(NULL, size, PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if(addr == MAP_FAILED)
	{
		//申请失败，内存不足
		printf("MallocBymmap申请失败\n");
		return NULL;
	}
	struct SLMemBlock *p = (struct SLMemBlock *)addr;

	p->CurBlkInUseBit = 1;
	p->PrevBlkInUseBit = 1;
	p->FromMmapBit = 1;
	p->ulBlockSize = size;
    cout<< p->ulBlockSize<<endl;
	return (char *)addr + sizeof(struct SLMemBlock);
}
int main()
{   
    struct SLMemBlock *pNextBlock = (struct SLMemBlock*)malloc(sizeof(struct SLMemBlock)*2);
    pNextBlock->CurBlkInUseBit = 1;
	pNextBlock->PrevBlkInUseBit = 1;
	pNextBlock->FromMmapBit = 1;
	pNextBlock->ulBlockSize = 32;
    cout<<pNextBlock<<endl;
    cout<<(unsigned long *)((char *)pNextBlock + pNextBlock->ulBlockSize)<<endl;
    cout<<((unsigned long *)((char *)pNextBlock + pNextBlock->ulBlockSize) - 1)<<endl;
    cout<<*((unsigned long *)((char *)pNextBlock + pNextBlock->ulBlockSize) - 1)<<endl;
	*((unsigned long *)((char *)pNextBlock + pNextBlock->ulBlockSize) - 1) = pNextBlock->ulBlockSize;
    cout<<*((unsigned long *)((char *)pNextBlock + pNextBlock->ulBlockSize) - 1)<<endl;
    // int size1 = 256 * 1024;
    // int size2 = 257 *1024;
    // int m = (size1)%(1024*4);
    // int n = (size1)/(1024*4);
    // int m2 = (size2)%(1024*4);
    // int n2 = (size2)/(1024*4);
    // int t1 = ((m == 0 )? n:(n+1));
    // int t2 = ((m2 == 0 )? n2:(n2+1));
    
    // cout<<"t1 "<<t1<<endl;
    // cout<<"t2 "<<t2<<endl;
    // cout<< ALIGN_UP_4KB(t1)<<endl;
    // cout<< ALIGN_UP_4KB(t2)<<endl;
    getchar();
    return 0;
}