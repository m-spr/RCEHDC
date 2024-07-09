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

ENTITY encoder IS
	GENERIC (d :  INTEGER := 500; 	-- dimension size 
			lgf  : INTEGER := 10;		 -- bit width out popCounters --- LOG2(#feature) --- nabayad in bashe!?! whats wrong with me?
			featureSize		: INTEGER := 700
			);	 
	PORT (
		clk, rst	: IN STD_LOGIC; 	-- run should be '1' for 2 clk cycle!!!!! badan behehesh fekr mikonam!!
		run			: IN  STD_LOGIC;
		din			: IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0);
		BV			: IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0);
		rundegi		: OUT STD_LOGIC;
		done, ready_M		: OUT  STD_LOGIC;
		dout		: OUT  STD_LOGIC_VECTOR (d-1 DOWNTO 0)
	);
END ENTITY encoder;

ARCHITECTURE behavioral OF encoder IS

component reg IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END component;

COMPONENT XoringInputPop IS
	GENERIC ( n  : INTEGER := 8		 -- number of loop
			 );	 
	PORT (
		clk, rst, update	: IN STD_LOGIC; 	--- bit width out popCounters --- LOG2(#feature)
		done	: IN  STD_LOGIC;
		din		: IN  STD_LOGIC;
		BV		: IN  STD_LOGIC;
		dout	: OUT  STD_LOGIC_VECTOR (n-1 DOWNTO 0)
	);
END COMPONENT;

COMPONENT  XoringPopCtrl IS
	GENERIC (n  : INTEGER := 10 ;			----bit width out popCounters --- ceiling log2(#feature)
			featureSize		: INTEGER := 700  );	---- NUmber of loops for popcounts 
	PORT (
		clk, rst 				: IN STD_LOGIC;
		run		 				: IN STD_LOGIC;
		rundegi, update, doneI, doneII, ready_M		    : OUT STD_LOGIC
	);
END COMPONENT;

signal update, doneI : STD_LOGIC;
CONSTANT test : STD_LOGIC_VECTOR (lgf-1 downto 0) := STD_LOGIC_VECTOR(to_UNSIGNED(featureSize/2 , lgf));
signal douttest	:  STD_LOGIC_VECTOR (d-1 DOWNTO 0);
signal doutXOR : STD_LOGIC_VECTOR ((d*lgf)-1 downto 0);
signal querycheck : STD_LOGIC_VECTOR ((d)-1 downto 0);
signal testID : STD_LOGIC_VECTOR (1000-1 DOWNTO 0) := (others=>'0');
signal testBV : STD_LOGIC_VECTOR (1000-1 DOWNTO 0) := (others=>'0');
SIGNAL IDeq, BVeq, eq : STD_LOGIC;

attribute MARK_DEBUG : string;
attribute MARK_DEBUG of doutXOR : signal is "TRUE";


BEGIN


	popCounters : FOR I IN 0 TO d-1 GENERATE
		pop: XoringInputPop 
			GENERIC MAP(lgf)
			PORT MAP(
				clk , rst, update, doneI,
				din(I), BV(I),
				doutXOR((I+1)*lgf-1 DOWNTO (I)*lgf)
			);
	END GENERATE popCounters;	

	ctrl :  XoringPopCtrl 
	GENERIC MAP(lgf, featureSize)
	PORT MAP(
		clk, rst, run,rundegi,
		update, doneI, done, ready_M
		);
	
	doutGen : FOR I IN 0 TO d-1 GENERATE
			dout (I) <= '1' WHEN ( doutXOR((I+1)*lgf-1 DOWNTO (I)*lgf) > test) ELSE '0';
	END GENERATE doutGen;
	


END ARCHITECTURE behavioral;
