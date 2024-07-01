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
    --dout(0) <= din(to_integer(unsigned(pointer)));
    --dout(1) <= din(to_integer(unsigned(pointer))+512);
	concat: FOR I IN 0 TO adI-1 GENERATE
		mm(I) <= din(to_integer(((2**n)*I) + unsigned(pointer)));
	END GENERATE concat;

dout <= mm;

END ARCHITECTURE;
