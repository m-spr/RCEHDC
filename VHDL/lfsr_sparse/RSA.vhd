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

ENTITY RSA IS
	GENERIC (n : INTEGER := 8;		 -- bit width out popCounters
			 d		 : INTEGER := 8;		 -- d,,, #popCounters
			 z		 : INTEGER := 0;		 -- zeropadding to 2**
			 logn : INTEGER := 3	);   -- MuxCell, ceilingLOG2(#popCounters)
	PORT (
		clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
		muxSel    : IN  STD_LOGIC_VECTOR (logn DOWNTO 0);
		A         : IN  STD_LOGIC_VECTOR (((d)*n)- 1 DOWNTO 0);  -- cascade with enough 0 as input or inner signal! lets check!
		B         : OUT  STD_LOGIC_VECTOR (n + logn - 1 DOWNTO 0)
	);
END ENTITY RSA;

ARCHITECTURE behavioral OF RSA IS
	Component recMux IS
		GENERIC ( n : INTEGER := 3;			-- MUX sel bit width (number of layer)
				  lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			sel 		: IN  STD_LOGIC_VECTOR (n DOWNTO 0);       -- sel<='0'&&sel
			din			: IN  STD_LOGIC_VECTOR (((2**n)*lenPop)- 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		); 
	END COMPONENT;
	Component SeqAdder IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
			din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop -1 DOWNTO 0)
		);
	END COMPONENT;
	CONSTANT zero_muxin : STD_LOGIC_VECTOR (((z+1)*n) -1 DOWNTO 0) := (OTHERS => '0');
	CONSTANT len : Integer := n+logn;
	CONSTANT zero : STD_LOGIC_VECTOR (logn -1 DOWNTO 0) := (OTHERS => '0');
	SIGNAL muxOut : STD_LOGIC_VECTOR (n -1 DOWNTO 0);
	SIGNAL muxIn : STD_LOGIC_VECTOR (((2**logn)+1)*n -1 DOWNTO 0);
	SIGNAL add : STD_LOGIC_VECTOR (len -1 DOWNTO 0);
BEGIN
	
	muxIn <= A & zero_muxin;
	m : recMux 
		GENERIC MAP(logn , n	
				   )
		PORT MAP(
			muxSel, muxIn((((2**logn)+1)*n) -1 DOWNTO n), muxOut
		);
	
	add <= zero & muxOut;
	
	addM : SeqAdder
		GENERIC MAP(len)
		PORT MAP(
			clk, rst, reg1Update, reg1rst, reg2Update, reg2rst , add, B 
		);
END ARCHITECTURE behavioral;