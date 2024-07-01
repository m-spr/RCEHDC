LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY regOne IS
	GENERIC (init : STD_LOGIC := '1');   -- initial value
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC;
		dout        : OUT  STD_LOGIC
	);
END ENTITY regOne;

ARCHITECTURE behavioral OF regOne IS
	SIGNAL regOut : STD_LOGIC;
BEGIN
	PROCESS(clk)
		BEGIN 
		    IF (clk ='1' and clk'event)THEN
			    IF(regrst = '1')THEN
				   regOut <= init;
				ELSIF (regUpdate ='1') THEN 
				   regOut <= din;
				END IF;
			END IF;
	END PROCESS;
	dout <= regOut;
END ARCHITECTURE behavioral; 