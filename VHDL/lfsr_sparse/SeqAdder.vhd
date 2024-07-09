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

ENTITY SeqAdder IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END ENTITY SeqAdder;

ARCHITECTURE behavioral OF SeqAdder IS
	SIGNAL addToReg, addToAdder : STD_LOGIC_VECTOR (lenPop- 1 DOWNTO 0);
	COMPONENT reg IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk 	: IN STD_LOGIC;
			regUpdate, regrst 	: IN STD_LOGIC;
			din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		);
	END COMPONENT;
BEGIN
	
	reg1 : reg 
		GENERIC MAP (lenPop)
		PORT MAP (clk, reg1Update, reg1rst, addToReg, addToAdder);
	reg2 : reg 
		GENERIC MAP (lenPop)
		PORT MAP (clk, reg2Update, reg2rst, addToAdder, dout);
		
	addToReg <= STD_LOGIC_VECTOR (UNSIGNED(din) + UNSIGNED(addToAdder));
END ARCHITECTURE behavioral;