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