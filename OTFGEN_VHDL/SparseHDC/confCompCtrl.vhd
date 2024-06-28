LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY confCompCtrl IS
	GENERIC (n : INTEGER := 10 ; 	-- #Classes
			 lgn : INTEGER := 4 );
	PORT (
		clk, rst 				: IN STD_LOGIC;
		run		 				: IN STD_LOGIC;
		runOut, done 			: OUT STD_LOGIC;
		pointer 				: OUT STD_LOGIC_VECTOR(lgn-1 DOWNTO 0) --- As of now only support up to 16 classes so 4'bits 
	);
END ENTITY confCompCtrl;

ARCHITECTURE ctrl OF confCompCtrl IS
	COMPONENT popCount IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk , rst 	: IN STD_LOGIC;
			en		 	: IN STD_LOGIC;
			dout        : OUT  STD_LOGIC_VECTOR (lenPop-1 DOWNTO 0)
		);
	END COMPONENT;
	SIGNAL count : STD_LOGIC_VECTOR (lgn-1 DOWNTO 0);
	TYPE state IS  (init,  counting);
	SIGNAL ns,  ps : state;
	SIGNAL countEn , countRst : STD_LOGIC;
BEGIN

	PROCESS(clk) BEGIN 
		IF rising_edge(clk) then
			IF (rst ='1')then
				ps <= init; 
			ELSE  
				ps <= ns;  
			END IF;
		END IF;
	END PROCESS;
	
	PROCESS ( ps,  run, count)
	BEGIN 
	runOut <= '0';
	countRst <= '0';
	countEn <= '0';	
    done <= '0';
		CASE (ps) IS 
			WHEN init =>
				countRst <= '1';
				runOut<= '1';
				IF ( run = '1') THEN
					ns <= counting;
				ELSE
					ns <= init;
				END IF;
			WHEN counting =>
				IF (count = STD_LOGIC_VECTOR(to_UNSIGNED(n , lgn))) THEN  --- perhaps -1 is extra! check
					done <= '1';
					IF ( run = '1') THEN
						countRst <= '1';
						runOut<= '1';
						ns <= counting;
					ELSE
						ns <= init;
					END IF;
				ELSE 
					countEn <= '1';	
					ns <= counting;
				END IF;
			WHEN OTHERS =>
					ns <= init;
		END CASE;
	END PROCESS;
	
	sel : popCount 
		GENERIC MAP(lgn)
		PORT MAP(
			clk, countRst, countEn, count
		);
	pointer <= count;

END ARCHITECTURE;	
