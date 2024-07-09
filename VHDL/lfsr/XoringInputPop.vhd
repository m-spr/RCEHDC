-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

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

attribute MARK_DEBUG : string;
attribute MARK_DEBUG of doutI : signal is "TRUE";
attribute MARK_DEBUG of XORR : signal is "TRUE";
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
