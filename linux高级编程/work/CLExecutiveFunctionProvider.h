#ifndef CLEXECUTIVEFUNCTIONPROVIDER_H
#define CLEXECUTIVEFUNCTIONPROVIDER_H
class CLExecutiveFunctionProvider
{
public:
	CLExecutiveFunctionProvider();
	virtual ~CLExecutiveFunctionProvider();

public:
	virtual void RunExecutiveFunction(void* pContext) = 0;

private:

};
#endif