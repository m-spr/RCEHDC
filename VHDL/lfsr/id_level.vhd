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
use std.textio.all; 

entity idLevel3 is
    GENERIC (n : INTEGER := 7;		 		 	-- bit-widths input
			 c 	: INTEGER := 2; 				-- coeficient of increasment! c = hv/ 2**n
			 r  : INTEGER := 2;                  -- remainder from division
			 hv : INTEGER := 500		 	 	-- hyperdimesional size
			 );
	Port ( values : in STD_LOGIC_VECTOR (n-1 DOWNTO 0);
           idVector : out STD_LOGIC_VECTOR (hv-1 DOWNTO 0));
end idLevel3;

architecture Behavioral of idLevel3 is

CONSTANT zeros : std_logic_vector (hv-1 DOWNTO 0) := (others => '0');
CONSTANT ones : std_logic_vector (hv-1 DOWNTO 0) := (others => '1');
SIGNAL idVectorVarS : STD_logic_vector ((2**n)-1 DOWNTO 0);
begin




PROCESS (values)
VARIABLE idVectorVar : std_logic_vector ((2**n)-1 DOWNTO 0);
begin
	   idVectorVar := (others => '0');
	if values = zeros(n-1 DOWNTO 0) then
		idVectorVar := (others => '0');
	elsif values = ones(n-1 DOWNTO 0) then
		idVectorVar := (others => '1');
	else
		idVectorVar(to_integer(unsigned(values))-1 DOWNTO 0) := ones(to_integer(unsigned(values))-1 DOWNTO 0);
	end if;
idVectorVarS <= idVectorVar;	
end PROCESS;

 ID_LOOP: FOR I IN C DOWNTO 1 GENERATE
		idVector(((2**n)*I)-1 Downto (2**n)*(I-1)) <= idVectorVarS;
	END GENERATE ID_LOOP;
	
remainder_ID : if r /= 0 generate
    idVector(hv-1 Downto (2**n)*(c)) <= idVectorVarS( r-1  Downto  0);
  end generate remainder_ID;


end Behavioral;
