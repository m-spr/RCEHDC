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
