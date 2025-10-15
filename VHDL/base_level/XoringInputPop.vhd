LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY XoringInputPop IS
	GENERIC ( n  : INTEGER := 8		 -- number of loop
			 );	 
	PORT (
		clk, rst, update	: IN STD_LOGIC; 	-- run should be '1' for 2 clk cycle!!!!! badan behehesh fekr mikonam!!
		done	: IN  STD_LOGIC;
		din		: IN  STD_LOGIC;
		BV		: IN  STD_LOGIC;
		dout	: OUT  STD_LOGIC_VECTOR (n-1 DOWNTO 0)
	);
END ENTITY XoringInputPop;

ARCHITECTURE behavioral OF XoringInputPop IS

COMPONENT popCount IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters --- LOG2(#feature)
	PORT (
		clk , rst 	: IN STD_LOGIC;
		en		 	: IN STD_LOGIC;
		dout        : OUT  STD_LOGIC_VECTOR (lenPop-1 DOWNTO 0)
	);
END COMPONENT;

COMPONENT reg IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END COMPONENT;

SIGNAL XORR, poprst : STD_LOGIC;
SIGNAL doutI	:  STD_LOGIC_VECTOR (n-1 DOWNTO 0);

BEGIN

	pop : popCount
	GENERIC MAP(n)   -- bit width out popCounters
	PORT MAP(
		clk , poprst,
		XORR,
		doutI
	);
	
	XORR <= (din XNOR BV) AND update;
	poprst <= rst or done;
	
	outreg : reg 
	GENERIC MAP(n)   -- bit width out popCounters
	PORT MAP(
		clk , done, rst , 
		doutI,
		dout 
	);
	
END ARCHITECTURE;
