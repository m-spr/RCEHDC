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

begin 
    --dout(0) <= din(to_integer(unsigned(pointer)));
    --dout(1) <= din(to_integer(unsigned(pointer))+512);
	concat: FOR I IN adI-1 DOWNTO 0 GENERATE
		dout(I) <= din(to_integer(unsigned(pointer))+((2**n)*I));
	END GENERATE concat;

END ARCHITECTURE;
