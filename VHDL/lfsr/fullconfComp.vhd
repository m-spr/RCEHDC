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

ENTITY fullconfComp  IS
	GENERIC (n : INTEGER := 10;		 --; 	-- bit--widths of memory pointer, counter and etc,,, 
		classNumber : INTEGER := 10; 		---- class number --- for memory image
		classPortion : INTEGER := 10 ); 		---- portion of class memory --- for memory image
	PORT (
		clk, rst, run, done  	: IN STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ---- 
		hv        		: IN  STD_LOGIC;
		Chv_input       		: IN  STD_LOGIC; --_vector ((2**n)-1 DOWNTO 0);
		pointer		 	: IN STD_LOGIC_VECTOR(n-1 DOWNTO 0);	
		sim		 	: OUT  STD_LOGIC_VECTOR(n-1 DOWNTO 0)  	
	);
END ENTITY fullconfComp ;

ARCHITECTURE behavioral OF fullconfComp IS
COMPONENT popCount IS
	GENERIC ( lenPop : INTEGER := 8 );   -- bit width out popCounters
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

SIGNAL toPop, popRST, regRST, memOut : STD_LOGIC;
SIGNAL count : STD_LOGIC_VECTOR (n-1 DOWNTO 0);

attribute MARK_DEBUG : string;
attribute MARK_DEBUG of Chv_input : signal is "TRUE";
BEGIN

	toPop <= (Chv_input XOR hv) and run ;
	popRST <= rst OR done;
	regRST <= rst OR run;
	
	pop: popCount 
	GENERIC MAP( n )
	PORT MAP(
		clk , popRST, toPop,
		count
	);
	
	regPop: reg 
	GENERIC MAP(n)
	PORT MAP(
		clk, 
		done, regRST,
		count,
		sim 
	);
	
END ARCHITECTURE behavioral;
