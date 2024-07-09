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

ENTITY hvTOcompIn IS
    GENERIC 
    (
        d           : INTEGER := 100000; -- dimension size
        n           : INTEGER := 7; -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,,
        adI         : INTEGER := 20	 -- number of confComp module, or adderInput and = ceiling(D/(2^n))
	); 
    PORT 
    (
        clk		   : IN STD_LOGIC;
        rst		   : IN STD_LOGIC; 
        din        : IN STD_LOGIC_VECTOR (d - 1 DOWNTO 0);
        pointer    : IN STD_LOGIC_VECTOR (n - 1 DOWNTO 0);
        dout	   : OUT STD_LOGIC_VECTOR(adI - 1 DOWNTO 0)
    );
END ENTITY hvTOcompIn;

ARCHITECTURE behavioral OF hvTOcompIn IS

SIGNAL mm : std_logic_vector(adI -1 DOWNTO 0);
--attribute mark_debug : string;
--attribute mark_debug of mm : signal is "TRUE";
--attribute mark_debug of din : signal is "TRUE";
--attribute mark_debug of dout : signal is "TRUE";

begin 
	concat: FOR I IN 0 TO adI-1 GENERATE
		mm(I) <= din(to_integer(((2**n)*I) + unsigned(pointer)));
	END GENERATE concat;

dout <= mm;

END ARCHITECTURE;
