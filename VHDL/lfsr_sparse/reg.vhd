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

ENTITY reg IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END ENTITY reg;

ARCHITECTURE behavioral OF reg IS
	SIGNAL regOut : STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
BEGIN
	PROCESS(clk)
		BEGIN 
		    IF (clk ='1' and clk'event)THEN
			    IF(regrst = '1')THEN
				   regOut <= (OTHERS=>'0');
				ELSIF (regUpdate ='1') THEN 
				   regOut <= din;
				END IF;
			END IF;
	END PROCESS;
	dout <= regOut;
END ARCHITECTURE behavioral;


LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY reg1 IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END ENTITY reg1;

ARCHITECTURE behavioral OF reg1 IS
	SIGNAL regOut : STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
BEGIN
	PROCESS(clk)
		BEGIN 
		    IF (clk ='1' and clk'event)THEN
			    IF(regrst = '1')THEN
				   regOut <= (OTHERS=>'1');
				ELSIF (regUpdate ='1') THEN 
				   regOut <= din;
				END IF;
			END IF;
	END PROCESS;
	dout <= regOut;
END ARCHITECTURE behavioral;