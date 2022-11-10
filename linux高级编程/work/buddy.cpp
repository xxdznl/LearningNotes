#include "buddy.h"
//是否存在空闲buddy内存块 =i意味着存在空闲块，根据申请大小判断，直接从对应大小层往上找
int IsListFreeEmpty(int size)
{   
    //对所有访问ListFree或者ListAlloc的地方上读锁
	//locker->lockRead();
    for(int i = BELONG_TO_N_LAYER(size);i<=14;i++)
    {
        //如果当前层链表为空，继续往上找,否则说明有合适大小的空闲块
        if(ListFree[i]->head->next == nullptr)
            continue;
        else
            return i;
    }
    //locker->unlockRead();
    //所有层都没有合适的空闲块
    return 0;
}
//空闲buddy内存块不够（初次申请内存，或者多次分配后无可用内存），找不到空闲buddy内存块。
//mmap一个2M大小的空间，并拆分为8个256kb，buddy块，插入ListFree[14] 额 不如直接mmap8个256kb的块，偷懒了~~~
void BuddyBlockAllocMmap2MB()
{
    //对ListFree修改了，上写锁
    //locker->lockWrite();
    Memory* temp = ListFree[14]->head;
    for(int i =0 ;i<8;i++)
    {
        Memory *p = MallocBymmap(THRESHOLD_FOR_MMAP);
        p->FromMmapBit = 0;//此时不能算是mmap直接分配
        temp->next = p;
        p->pre = temp;
        temp = p;
    }
    //locker->unlockWrite();
    return ;
    
};
//使用buddy算法分配空闲块, 输入需求的大小。
Memory* BuddyAllocBlock(size_t size)
{
    //直接去第layer层找,前面已经判断过必定有合适大小的块,直接取出当前层第一块
    int layer = IsListFreeEmpty(size);
    //分块过程直接写锁
    //locker->lockWrite();    
    //locker->unlockWrite();
    //把target卸下来，分块后删除
    //删除块修改了三个块，当前块，前一块，后一块，都加锁
    Memory* target =  ListFree[layer]->head->next;
    void* address = target->address;
    //if(pthread_mutex_trylock(&(ListFree[layer]->head)->mutex)==0 && pthread_mutex_trylock(&(target)->mutex)==0)
    {
        ListFree[layer]->head->next = target->next;
        //还剩下不止一块时，下一块指向头结点
        if(target->next !=nullptr) //&& (pthread_mutex_trylock(&(target->next)->mutex)==0))//在别的线程可能会修改target（即ListFree[layer]->head->next）的值，因此也要加锁
        {
            target->next->pre = ListFree[layer]->head;
            //pthread_mutex_unlock(&(target->next)->mutex);
        }
        target->next = nullptr;
        target->pre = nullptr;
        //pthread_mutex_unlock(&(ListFree[layer]->head)->mutex);
        //pthread_mutex_unlock(&(target)->mutex);
    }
    //删除一开始的target结点
    delete target;
    //申请大小本应在第origin_layer层，现在跨了layer - originLayer层，向下分块。
    //比如14层是256kb 12层是64kb，跨了两层，向下分两次块
    int originLayer = BELONG_TO_N_LAYER(size);
    Memory * blockDivided = nullptr;
    Memory* blockLeft= nullptr;
    for(int i = layer-1; i >= originLayer;i--)
    {   
        //分成两块后剩下的一块，连在当前层的头。另一块继续分块
        blockLeft = new Memory(address,ListFree[i]->size);
        //第2块地址
        address = (char*)address + ListFree[i]->size;
        //第一块连接本层
        //插入操作修改了两个块 插入块前一块和后一块，两块都要加锁，插入块当前环境下新创建的，唯一不用加锁
        //if((pthread_mutex_trylock(&(ListFree[i]->head->next)->mutex)==0 && (pthread_mutex_trylock(&(ListFree[i]->head)->mutex)==0)))
        {

            blockLeft->next = ListFree[i]->head->next;
            //当前层不为空
            if(ListFree[i]->head->next !=nullptr)
                ListFree[i]->head->next->pre = blockLeft;
            ListFree[i]->head->next=blockLeft;
            blockLeft->pre = ListFree[i]->head;
            //pthread_mutex_unlock(&(ListFree[i]->head->next)->mutex);
            //pthread_mutex_unlock(&(ListFree[i]->head)->mutex);
        }
    }

    //0 0   循环1 块起始地址0   大小128  下一块起始地址addr = 128 
    //      循环2 块起始地址128 大小64   下一块起始地址addr = 192 
    //循环结束后,将最终另一块分配出去
    blockDivided = new Memory(address,ListFree[originLayer]->size);
    
    return blockDivided;
};
//合并到空闲块链表
int BuddyMergeToFreeList(Memory *MemoryToMerge)
{
    //递归调用这部分
	//根据块大小判断是第几层的块
    int layer = BELONG_TO_N_LAYER(MemoryToMerge->size);
    //判断该层是否有兄弟块
    int flag = 0;
    //合并过程直接写锁
    //locker->lockWrite();
    Memory * temp = ListFree[layer]->head;
    while(temp->next != nullptr)
    {
        temp = temp->next;
        long x = (char *)(MemoryToMerge->address)-(char *)(temp->address);
        //std::cout<<"块1-块2"<<x<<std::endl;
        if(((x < 0)&& (x+ListFree[layer]->size) == 0 )||((x > 0)&& (x-ListFree[layer]->size) == 0))
        {
            pthread_mutex_lock(coutMutex);
            std::cout<<"|----第"<<layer<<"层有兄弟块"<<std::endl;
            std::cout<<"|----兄弟块1地址"<<MemoryToMerge->address<<" 大小为："<<MemoryToMerge->size<<"B|"<<std::endl<<"|----兄弟块2地址"<<temp->address<<" 大小为："<<temp->size<<"B|"<<std::endl<<std::endl;
            //std::cout<<"块ToMerge-块temp "<<x<<std::endl;
            pthread_mutex_unlock(coutMutex);
        }
        if((x < 0)&& (x+ListFree[layer]->size) == 0)
        {
            pthread_mutex_lock(coutMutex);
            //std::cout<<"块1地址"<<MemoryToMerge->address<<" 大小为："<<MemoryToMerge->size<<std::endl<<"块2地址"<<temp->address<<" 大小为："<<temp->size<<std::endl<<std::endl;
            std::cout<<"MemoryToMerge低位"<<std::endl;
            pthread_mutex_unlock(coutMutex);
            flag = 1;
            break;
        }
        //temp是低地址块
        else if((x > 0)&& (x-ListFree[layer]->size) == 0)
        {
            // pthread_mutex_lock(coutMutex);
            // std::cout<<"temp低位"<<x<<std::endl<<std::endl;
            // pthread_mutex_unlock(coutMutex);
            flag = 2;
            break;
        }
    }
    //加锁temp当前块，
    //if(pthread_mutex_trylock(&(temp)->mutex)==0)
    {
        //没有，直接头插，插入该层，返回
        //或者到第14层也应该直接插入
        if(flag == 0||layer ==14)
        {
            pthread_mutex_lock(coutMutex);
            std::cout<<"|----最后插入了第"<<layer<<"层, 层大小为"<<ListFree[layer]->size<<"B|"<<std::endl;
            pthread_mutex_unlock(coutMutex);
            //MemoryToMerge是低地址块
            Memory * temp = ListFree[layer]->head;
            //插入操作修改了两个块 插入块前一块和后一块，两块都要加锁，插入块上一层传过来的，唯一不用加锁
            //if((pthread_mutex_trylock(&(temp->next->pre)->mutex)==0))
            {

                MemoryToMerge->next = temp->next;
                if(temp->next !=nullptr)
                    temp->next->pre = MemoryToMerge;
                temp->next = MemoryToMerge;
                MemoryToMerge->pre = temp;
                //pthread_mutex_unlock(&(temp)->mutex);
                //pthread_mutex_unlock(&(temp->next->pre)->mutex);
            }
            //locker->unlockWrite();
            return layer;
        }
        //有兄弟块，保留低地址块结点，删除高地址块结点，修改低地址块结点大小，将其插入下一层
        else 
        {
            //不论temp是不是低地址块都要先把temp从layer层卸下来
            Memory * tempPre =temp->pre;
            //删除块修改了三个块，当前块，前一块，后一块，都加锁
            //std::cout<<"有兄弟块  ";
            //if(pthread_mutex_trylock(&(tempPre)->mutex)==0)
            {
                tempPre->next = temp->next;
                //还剩下不止一块时，下一块指向头结点
                if(temp->next != nullptr )//&& (pthread_mutex_trylock(&(temp->next)->mutex)==0))//在别的线程可能会修改target（即ListFree[layer]->head->next）的值，因此也要加锁
                {
                    temp->next->pre = tempPre;
                    //pthread_mutex_unlock(&(temp->next)->mutex);
                }
                temp->pre = nullptr;
                temp->next = nullptr;
                //pthread_mutex_unlock(&(tempPre)->mutex);
            }
            
            pthread_mutex_unlock(&(temp)->mutex);
            //(flag == 1) MemoryToMerge是低地址块,，后将MemoryToMerg大小扩大后送入下一层
            if(flag == 1)
            {
                
                MemoryToMerge->size += temp->size;
                delete temp;
                return BuddyMergeToFreeList(MemoryToMerge);
            }
            //(flag == 2) temp是低地址块，大小扩大后送入下一层，删除MemoryToMerge
            else if(flag == 2)
            {
                temp->size += MemoryToMerge->size;
                delete MemoryToMerge;
                return BuddyMergeToFreeList(temp);
            }
        }
    }
    return -1;
};