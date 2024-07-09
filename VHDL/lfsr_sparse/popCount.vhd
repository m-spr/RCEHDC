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

ENTITY popCount IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters --- LOG2(#feature)
	PORT (
		clk , rst 	: IN STD_LOGIC;
		en		 	: IN STD_LOGIC;
		dout        : OUT  STD_LOGIC_VECTOR (lenPop-1 DOWNTO 0)
	);
END ENTITY popCount;

ARCHITECTURE behavioral OF popCount IS
SIGNAL popOut : STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
	
BEGIN

	PROCESS(clk)
		BEGIN 
		    IF (clk ='1' and clk'event)THEN
			    IF(rst = '1')THEN
				   popOut <= (OTHERS=>'0');
				ELSIF (en ='1') THEN 
				   popOut <= STD_LOGIC_VECTOR (UNSIGNED(popOut) + 1);
				END IF;
			END IF;
	END PROCESS;

	dout <= popOut;
END ARCHITECTURE behavioral;
